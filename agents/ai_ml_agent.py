from agents.base_agent import BaseAgent

class AiMlAgent(BaseAgent):
    SYSTEM = """You are an expert in Python AI/ML/DL systems.
Domains: PyTorch, HuggingFace Transformers, LangChain, LangGraph, RAG pipelines,
LoRA fine-tuning, SLM architecture, ChromaDB, FAISS, Ollama integration.
Always: add type hints, handle GPU/CPU fallback, include memory optimization,
add progress logging, write modular reusable code."""

    def __init__(self, model, manager, tokenizer):
        super().__init__(model, manager, "ai_ml", self.SYSTEM, tokenizer)

    def run(self, instruction: str, context: dict = None) -> dict:
        result = super().run(instruction, context)
        result["type"] = "ai_ml_code"
        return result
