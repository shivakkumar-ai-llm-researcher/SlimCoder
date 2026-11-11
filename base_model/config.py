from dataclasses import dataclass

@dataclass
class ShivCoderConfig:
    vocab_size: int = 32000
    block_size: int = 1024  # Max sequence length. WARNING: Limits single-file generation to ~750 tokens of output after prompt overhead.
    n_layer: int = 4        # Keeping it lightweight for CPU execution
    n_head: int = 4
    n_embd: int = 256
    bias: bool = True
    dropout: float = 0.0
