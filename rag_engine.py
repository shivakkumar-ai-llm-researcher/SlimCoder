import os
import glob
import torch
import torch.nn.functional as F
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class LocalRAG:
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = workspace_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.documents = []
        self.embeddings = None
        self.file_paths = []
        self.is_ready = False

        if SentenceTransformer:
            try:
                # Load a lightweight embedding model
                print("[RAG] Loading embedding model 'all-MiniLM-L6-v2'...")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                self.is_ready = True
            except Exception as e:
                print(f"[RAG] Failed to load embedding model: {e}")
        else:
            print("[RAG] sentence_transformers not installed. RAG will be disabled.")

    def index_workspace(self, target_dir: str = None):
        """Indexes the files in the workspace to create a parameterless memory store."""
        if not self.is_ready:
            return

        search_dir = target_dir if target_dir else self.workspace_path
        print(f"[RAG] Indexing workspace: {search_dir}")
        
        # extensions to index (Must stay in sync with the adapter registry in agents/orchestrator.py)
        exts = [".py", ".js", ".ts", ".tsx", ".dart", ".java", ".cs", ".kt", ".html", ".css", ".md"]
        
        self.documents = []
        self.file_paths = []

        for root, dirs, files in os.walk(search_dir):
            # Skip hidden directories and large folders like node_modules, venv
            if any(part.startswith('.') or part in ['node_modules', 'venv', '.venv', '__pycache__', 'checkpoints', 'adapters'] for part in root.split(os.sep)):
                continue

            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if len(content.strip()) > 0:
                                # For simplicity, treat the whole file as a single document/chunk
                                # In a real large codebase, we would chunk this
                                self.documents.append(content)
                                self.file_paths.append(file_path)
                    except Exception:
                        pass
        
        if len(self.documents) > 0:
            print(f"[RAG] Encoding {len(self.documents)} documents...")
            # Compute embeddings
            with torch.no_grad():
                self.embeddings = self.model.encode(self.documents, convert_to_tensor=True)
            print("[RAG] Indexing complete.")
        else:
            print("[RAG] No documents found to index.")

    def retrieve_context(self, query: str, top_k: int = 2) -> dict:
        """Searches the vector store and returns the most relevant files as context."""
        if not self.is_ready or self.embeddings is None or len(self.documents) == 0:
            return {}

        with torch.no_grad():
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            # Compute cosine similarities
            cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings, dim=-1)
            
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
            
        context = {}
        for score, idx in zip(top_results[0], top_results[1]):
            # Only include if score is above a basic threshold to avoid noise
            if score.item() > 0.1:
                path = self.file_paths[idx]
                rel_path = os.path.relpath(path, self.workspace_path)
                context[rel_path] = self.documents[idx]
                print(f"[RAG] Retrieved '{rel_path}' (score: {score.item():.4f})")
                
        return context

if __name__ == "__main__":
    # Test RAG Engine
    rag = LocalRAG()
    rag.index_workspace()
    res = rag.retrieve_context("login token endpoint")
    for k in res.keys():
        print("Found:", k)
