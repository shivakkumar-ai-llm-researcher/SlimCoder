# bootstrap_base.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from logger_config import setup_system_logger

logger = setup_system_logger()

def check_gpu_memory():
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / (1024 ** 2)
        return round(vram, 2)
    return 0.0

def run_layer_pruning():
    logger.info("INITIATING: Starting Local Layer Pruning protocol.")
    
    # 1. Target parent model optimized for code logic
    parent_model_id = "Qwen/Qwen2.5-Coder-1.5B"
    logger.info(f"FETCHING: Loading parent framework weights from {parent_model_id}")
    
    # Load teacher/parent model in float16 to match your RTX 2050 capabilities
    try:
        parent_model = AutoModelForCausalLM.from_pretrained(
            parent_model_id, 
            torch_dtype=torch.float16,
            device_map="cpu" # Load to CPU first to prevent straining 4GB VRAM during surgery
        )
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load parent model. Error: {str(e)}")
        return

    # 2. Select structural layers to preserve (Stitch uniform blocks across the architecture)
    # Original model has 28 layers. Selecting 12 layers slashes size to ~380M parameters.
    # RATIONALE: We apply a uniform stride sampling strategy across the 28 layers
    # (selecting early layers for low-level syntax representation, mid layers for logical semantics,
    # and late layers for language head context) to preserve representational breadth and prevent
    # cognitive collapse.
    selected_layers = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27]
    logger.info(f"STRATEGY: Selecting {len(selected_layers)} core layers to prevent cognitive collapse.")

    # 3. Clone configuration structure and adjust layers
    config = AutoConfig.from_pretrained(parent_model_id)
    config.num_hidden_layers = len(selected_layers)
    
    # Instantiate custom low-footprint base architecture shell
    logger.info("COMPILING: Initializing target lightweight base model structure...")
    custom_base_model = AutoModelForCausalLM.from_config(config)

    # 4. Surgical weight copying across matrices
    logger.info("SURGERY: Injecting parent coherence weights into custom base tensors...")
    with torch.no_grad():
        # Copy foundational components (Embeddings, Norms, Language Heads)
        custom_base_model.model.embed_tokens.weight.copy_(parent_model.model.embed_tokens.weight)
        custom_base_model.lm_head.weight.copy_(parent_model.lm_head.weight)
        if hasattr(parent_model.model, 'norm'):
            custom_base_model.model.norm.weight.copy_(parent_model.model.norm.weight)

        # Map structural code layers perfectly without breaks
        for new_index, old_index in enumerate(selected_layers):
            custom_base_model.model.layers[new_index].load_state_dict(
                parent_model.model.layers[old_index].state_dict()
            )
            logger.info(f"MAPPED: Layer block {old_index} (Parent) -> Layer block {new_index} (Custom Base)")

    # 5. Save the final compiled base model weights and config structure to checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    output_path = "checkpoints/base_pretrained.pt"
    
    logger.info(f"SAVING: Exporting structured base file to {output_path}")
    checkpoint_data = {
        "model": custom_base_model.state_dict(),
        "config": config
    }
    torch.save(checkpoint_data, output_path)
    
    # Log successful execution parameters
    total_params = custom_base_model.num_parameters() / 1e6
    logger.info(f"SUCCESS: Custom Base Compiled! Total Parameters: {total_params:.2f} Million.")

if __name__ == "__main__":
    run_layer_pruning()
