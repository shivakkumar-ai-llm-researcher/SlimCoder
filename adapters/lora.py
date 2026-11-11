import os
import torch
import torch.nn as nn
from typing import Dict

class LoRALayer(nn.Module):
    """Low-Rank Adaptation — adds only a fraction of parameters per adapter"""
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # lora_A: down-projection (rank, in_features)
        # lora_B: up-projection (out_features, rank)
        # Only these parameters will be trained per domain
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A to normal distribution, B to zero (so adapter begins as identity)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x is (batch, seq, in_features)
        # Ensure parameters match input tensor dtype (prevents bfloat16/float32 mismatch)
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        return (x @ A.T @ B.T) * self.scale

class LoRALinear(nn.Module):
    """Wraps an existing Linear layer with LoRA — base weights are FROZEN"""
    def __init__(self, linear: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.linear = linear
        # Freeze base layer weights and bias
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
            
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        # Cast LoRA layer to linear layer's dtype
        self.lora.to(linear.weight.dtype)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class AdapterManager:
    """Hot-swaps LoRA adapters into the base model at runtime"""
    def __init__(self, base_model, device="cuda"):
        self.base_model = base_model
        self.device = device
        self.current_adapter = None
        self.adapters: Dict[str, dict] = {} # adapter_name -> state_dict
        self.is_merged = False
        
        # Save a copy of the base model weights (excluding lora parameters) to allow clean resets
        self.original_base_state = {
            k: v.clone() for k, v in base_model.state_dict().items()
            if "lora_" not in k
        }

        # Inject LoRA into QKV, projections, and MLP linear layers
        self._inject_lora_hooks(base_model)

    def _inject_lora_hooks(self, model, rank=16):
        """Replace linear layers in attention and MLP with LoRALinear wrappers"""
        modules_to_replace = []
        target_keys = ["c_attn", "c_proj", "c_fc", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(
                k in name for k in target_keys
            ):
                modules_to_replace.append((name, module))

        print(f"Injecting LoRA hooks: Replacing {len(modules_to_replace)} Linear layers with LoRALinear wrappers...")
        for name, module in modules_to_replace:
            parent = self._get_parent(model, name)
            attr = name.split(".")[-1]
            setattr(parent, attr, LoRALinear(module, rank=rank))

    def _get_parent(self, model, name):
        parts = name.split(".")
        for part in parts[:-1]:
            model = getattr(model, part)
        return model

    def load_adapter(self, adapter_name: str, adapter_path: str):
        """Load adapter weights from disk, or initialize empty zeros if file not found"""
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter file {adapter_path} not found! Initializing empty zero weights (identity mapping).")
            # Build an empty state matching all lora parameters in base_model
            state = {}
            for name, param in self.base_model.named_parameters():
                if "lora_" in name:
                    state[name] = torch.zeros_like(param.data)
        else:
            state = torch.load(adapter_path, map_location=self.device)
            
        self.adapters[adapter_name] = state
        print(f"Loaded adapter state dict: {adapter_name}")

    def activate(self, adapter_name: str):
        """Hot-swap to a different LoRA adapter (< 50ms). Bypassed if in locked merged model mode."""
        if self.is_merged:
            return
            
        if adapter_name == self.current_adapter:
            return
            
        if adapter_name not in self.adapters:
            # Try to load default from files
            path = f"adapters/{adapter_name}_lora.pt"
            self.load_adapter(adapter_name, path)

        state = self.adapters[adapter_name]
        # Load only lora_A and lora_B weights
        lora_state = {k: v for k, v in state.items() if "lora_" in k}
        self.base_model.load_state_dict(lora_state, strict=False)
        self.current_adapter = adapter_name
        print(f"Activated adapter: {adapter_name}")
    def merge_adapters_to_base(self, adapter_names: list):
        """
        Merge multiple LoRA adapters directly into the base model weights and disable active LoRA forwards.
        
        WARNING: This operation is destructive and in-memory only. The in-memory base model weights are mutated.
        The original checkpoints on disk are never overwritten. A snapshot of the merged session is saved to
        checkpoints/merged_session.pt for potential reuse.
        """
        # 1. Reset base model weights to original state dict to clear previous merges
        self.base_model.load_state_dict(self.original_base_state, strict=False)

        # 2. For each adapter, calculate delta and add it to base linear weights in-place
        for name in adapter_names:
            if not name:
                continue
            if name not in self.adapters:
                path = f"adapters/{name}_lora.pt"
                self.load_adapter(name, path)

            state = self.adapters[name]
            
            # Map LoRA weights to LoRALinear wrappers
            for mod_name, module in self.base_model.named_modules():
                if isinstance(module, LoRALinear):
                    prefix = f"{mod_name}.lora"
                    key_A = f"{prefix}.lora_A"
                    key_B = f"{prefix}.lora_B"
                    
                    if key_A in state and key_B in state:
                        A = state[key_A].to(self.device)
                        B = state[key_B].to(self.device)
                        scale = module.lora.scale
                        
                        # Delta W = (B @ A) * scale
                        delta = (B @ A) * scale
                        module.linear.weight.data.add_(delta)

        # 3. Zero out active LoRA parameters so they act as identity and do not double add
        for name, param in self.base_model.named_parameters():
            if "lora_" in name:
                param.data.zero_()

        self.current_adapter = f"merged_{'_'.join(filter(None, adapter_names))}"
        print(f"Successfully merged adapters {adapter_names} into base weights. New state: {self.current_adapter}")
        
        # 4. Save a snapshot of the merged weights for reference or debugging
        os.makedirs("checkpoints", exist_ok=True)
        merged_path = "checkpoints/merged_session.pt"
        torch.save(self.base_model.state_dict(), merged_path)
        print(f"Saved merged session snapshot to {merged_path}")


    def save_adapter(self, adapter_name: str, save_path: str):
        """Save only the LoRA weights (excluding the heavy base model parameters)"""
        lora_state = {
            k: v for k, v in self.base_model.state_dict().items()
            if "lora_" in k
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(lora_state, save_path)
        self.adapters[adapter_name] = lora_state
        print(f"Saved {len(lora_state)} LoRA tensors to -> {save_path}")
