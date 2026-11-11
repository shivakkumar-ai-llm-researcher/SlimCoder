import os
import sys
import json
import torch
import contextlib
import warnings

# Suppress FutureWarnings and RequestsDependencyWarning in log outputs
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except ImportError:
    pass

from adapters.lora import AdapterManager
from base_model.architecture import ShivCoder
from base_model.config import ShivCoderConfig
from tokenizers import Tokenizer

DOMAIN_CONFIGS = {
    "react": {
        "data":    "data/cache/react_corpus.jsonl",
        "system":  "You are an expert React developer. Use TypeScript, hooks, Zustand, TanStack Query, and Tailwind. Write production-quality component code.",
        "save_to": "adapters/react_lora.pt",
        "epochs":  3,
    },
    "react_native": {
        "data":    "data/cache/rn_corpus.jsonl",
        "system":  "You are an expert React Native + Expo developer. Use TypeScript, Expo Router, NativeWind, Zustand, and TanStack Query. Handle mobile UX patterns.",
        "save_to": "adapters/react_native_lora.pt",
        "epochs":  3,
    },
    "flutter": {
        "data":    "data/cache/flutter_corpus.jsonl",
        "system":  "You are an expert Flutter + Dart developer. Use Riverpod, GoRouter, HTTP/Dio. Write clean widgets, follow Dart style guide, write responsive layouts, and handle state management.",
        "save_to": "adapters/flutter_lora.pt",
        "epochs":  3,
    },
    "angular": {
        "data":    "data/cache/angular_corpus.jsonl",
        "system":  "You are an expert Angular developer. Use standalone components, RxJS, NgRx, HttpClient. Implement lifecycle hooks, forms, and subscriptions.",
        "save_to": "adapters/angular_lora.pt",
        "epochs":  3,
    },
    "nextjs": {
        "data":    "data/cache/nextjs_corpus.jsonl",
        "system":  "You are an expert Next.js developer. Use App Router, React 18, Server Actions, Tailwind CSS. Handle error boundaries and SSR/CSR optimization.",
        "save_to": "adapters/nextjs_lora.pt",
        "epochs":  3,
    },
    "python_backend": {
        "data":    "data/cache/python_corpus.jsonl",
        "system":  "You are an expert Django REST Framework developer. Write secure ViewSets, serializers, JWT auth, Celery tasks, and PostgreSQL queries.",
        "save_to": "adapters/python_backend_lora.pt",
        "epochs":  3,
    },
    "python_fastapi": {
        "data":    "data/cache/fastapi_corpus.jsonl",
        "system":  "You are an expert Python FastAPI developer. Use Pydantic v2, SQLAlchemy/SQLModel, dependency injection, async endpoints.",
        "save_to": "adapters/python_fastapi_lora.pt",
        "epochs":  3,
    },
    "java_springboot": {
        "data":    "data/cache/springboot_corpus.jsonl",
        "system":  "You are an expert Java Spring Boot developer. Use constructor injection, controller-service-repository, global exceptions, and Spring Security.",
        "save_to": "adapters/java_springboot_lora.pt",
        "epochs":  3,
    },
    "js_express": {
        "data":    "data/cache/express_corpus.jsonl",
        "system":  "You are an expert Node.js Express developer. Use modular routers, middleware, Joi/Zod validation, and RESTful routing patterns.",
        "save_to": "adapters/js_express_lora.pt",
        "epochs":  3,
    },
    "dotnet": {
        "data":    "data/cache/dotnet_corpus.jsonl",
        "system":  "You are an expert C# .NET Web API developer. Use async/await controllers, Entity Framework Core, Repository pattern, and Clean Architecture.",
        "save_to": "adapters/dotnet_lora.pt",
        "epochs":  3,
    },
    "ai_ml": {
        "data":    "data/cache/ai_ml_corpus.jsonl",
        "system":  "You are an expert in Python ML/DL/AI. You master PyTorch, HuggingFace, LangChain, LangGraph, RAG pipelines, LoRA fine-tuning, and multi-agent systems.",
        "save_to": "adapters/ai_ml_lora.pt",
        "epochs":  4,
    },
    "merger": {
        "data":    "data/cache/merger_corpus.jsonl",
        "system":  "You are a full-stack integration expert. Given frontend (React/RN) and backend (Django) code, align API contracts, resolve type mismatches, generate axios clients, and stitch both layers into working end-to-end code.",
        "save_to": "adapters/merger_lora.pt",
        "epochs":  4,
    },
}

def tokenize(text: str, tokenizer: Tokenizer, max_len: int = 1024) -> torch.Tensor:
    tokens = tokenizer.encode(text).ids
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)

def train_adapter(domain: str):
    if domain not in DOMAIN_CONFIGS:
        print(f"Error: Unknown domain '{domain}'. Choose from {list(DOMAIN_CONFIGS.keys())}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained base model config & weights
    base_checkpoint_path = "checkpoints/base_pretrained.pt"
    if not os.path.exists(base_checkpoint_path):
        print(f"Error: Base checkpoint not found at {base_checkpoint_path}. Run bootstrap_base.py first.")
        return

    base = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    
    # Safely extract state dict and config
    if isinstance(base, dict) and "model" in base:
        state_dict = base["model"]
    else:
        state_dict = base

    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", local_files_only=True)
        config.num_hidden_layers = 12
    except Exception:
        # Fallback offline config structure
        config = AutoConfig.from_dict({
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0
        })

    model = AutoModelForCausalLM.from_config(config)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Base Qwen model weights loaded successfully (strict=True) for training.")
    except Exception as e:
        model.load_state_dict(state_dict, strict=False)
        print(f"Warning: Base model loaded with strict=False for training: {e}")
    model = model.to(device)

    # Attach LoRA — freezes base, only trains adapters
    manager = AdapterManager(model, device)

    domain_cfg = DOMAIN_CONFIGS[domain]
    
    # Load tokenizer
    tokenizer_path = "base_model/tokenizer/shiv_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Run tokenizer_generator.py first.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Configure mixed precision
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()

    # Only optimize LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    if len(lora_params) == 0:
        print("Error: No LoRA trainable parameters found! Ensure LoRA hooks were injected properly.")
        return
        
    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)

    print(f"\nTraining adapter: {domain}")
    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params)/1e6:.4f}M")

    model.train()
    epochs = domain_cfg["epochs"]
    data_file = domain_cfg["data"]

    if not os.path.exists(data_file):
        print(f"Error: Training dataset not found at {data_file}. Run generate_mock_data.py first.")
        return

    for epoch in range(epochs):
        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                prompt = f"<|system|>{domain_cfg['system']}<|user|>{item['instruction']}<|assistant|>"
                target = item["output"]
                full = prompt + target + "<|end|>"

                # Tokenize inline
                max_len = getattr(config, "max_position_embeddings", 1024)
                if hasattr(config, "block_size"):
                    max_len = config.block_size
                x = tokenize(full, tokenizer, max_len=max_len).to(device)
                y = x.clone()

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                with ctx:
                    # Causal training: compute cross entropy loss manually from logits
                    outputs = model(x[:-1].unsqueeze(0))
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        y[1:].view(-1), 
                        ignore_index=-1
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if i % 2 == 0:
                    print(f"  Epoch {epoch} | Step {i} | Loss {loss.item():.4f}")

    # Save only the LoRA weights (excluding frozen base parameters)
    manager.save_adapter(domain, domain_cfg["save_to"])
    print(f"Completed training adapter for {domain} -> saved to {domain_cfg['save_to']}")

if __name__ == "__main__":
    domain_arg = sys.argv[1] if len(sys.argv) > 1 else "react"
    train_adapter(domain_arg)
