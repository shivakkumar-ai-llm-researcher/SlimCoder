import os
import requests
import torch
from tokenizers import Tokenizer

class CloudOrchestrator:
    def __init__(self):
        # Safely extract your key from the environment variables
        self.api_key = os.getenv("NVIDIA_API_KEY", "your_nvapi_key_here")
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        
    def call_kimi_orchestrator(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "model": "moonshotai/kimi-k2.6",
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are an automated code dataset compiler. Output raw JSON arrays. "
                        "CRITICAL: Do not describe the code, do not write conversational introductions, "
                        "and do not repeat words. Begin directly with the '[' character."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.4,       # Increased from 0.2 to explicitly stop the "the, the, the" repetition loop
            "top_p": 0.9,
            "response_format": { "type": "json_object" } # Force-aligns the underlying NIM engine to valid JSON outputs
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Fallback Triggered. Cloud API Error: {str(e)}"
from adapters.lora import AdapterManager
from base_model.architecture import ShivCoder
from base_model.config import ShivCoderConfig
from agents.orchestrator import Orchestrator, Domain
from agents.react_agent import ReactAgent
from agents.react_native_agent import ReactNativeAgent
from agents.python_agent import PythonBackendAgent
from agents.ai_ml_agent import AiMlAgent
from agents.merger_agent import MergerAgent
from rag_engine import LocalRAG

class OrgSLMPipeline:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Initializing on device: {self.device}")

        # Load shared base model (remains in RAM/VRAM)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Base checkpoint not found at {checkpoint_path}. Please run bootstrap_base.py first.")

        base_checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Safely extract state dict and config
        saved_config = None
        if isinstance(base_checkpoint, dict) and "model" in base_checkpoint:
            state_dict = base_checkpoint["model"]
            saved_config = base_checkpoint.get("config")
        else:
            state_dict = base_checkpoint

        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Load configuration structure for the 12-layer pruned base model
        if saved_config is not None:
            cfg_dict = saved_config.to_dict() if hasattr(saved_config, "to_dict") else dict(saved_config)
            model_type = cfg_dict.pop("model_type", "qwen2")
            num_layers = cfg_dict.get("num_hidden_layers", 12)
            if isinstance(cfg_dict.get("layer_types"), list) and len(cfg_dict["layer_types"]) != num_layers:
                cfg_dict["layer_types"] = cfg_dict["layer_types"][:num_layers]
            self.config = AutoConfig.for_model(model_type, **cfg_dict)
        else:
            try:
                self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", local_files_only=True)
                self.config.num_hidden_layers = 12
            except Exception:
                # Fallback configuration structure if offline
                self.config = AutoConfig.for_model(
                    "qwen2",
                    vocab_size=151936,
                    hidden_size=1536,
                    intermediate_size=8960,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_theta=1000000.0
                )
            
        target_dtype = next(iter(state_dict.values())).dtype if state_dict else torch.bfloat16
        self.model = AutoModelForCausalLM.from_config(self.config, dtype=target_dtype).to(self.device)
        
        # Load state dict strictly matching the compiled Qwen configuration
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("[Pipeline] Upgraded pruned 795M Qwen base model loaded successfully.")
        except Exception as e:
            # Safe fallback if there is any mismatch
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[Pipeline] Warning: Base model loaded with strict=False: {e}")
        self.model.eval()

        # Load tokenizer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(current_dir, "base_model", "tokenizer", "shiv_tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please run tokenizer_generator.py first.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Attach LoRA manager and load all adapters into memory
        self.manager = AdapterManager(self.model, self.device)
        
        adapter_files = {
            "react":          os.path.join(current_dir, "adapters", "react_lora.pt"),
            "react_native":   os.path.join(current_dir, "adapters", "react_native_lora.pt"),
            "flutter":        os.path.join(current_dir, "adapters", "flutter_lora.pt"),
            "angular":        os.path.join(current_dir, "adapters", "angular_lora.pt"),
            "nextjs":         os.path.join(current_dir, "adapters", "nextjs_lora.pt"),
            "python_backend": os.path.join(current_dir, "adapters", "python_backend_lora.pt"),
            "python_fastapi": os.path.join(current_dir, "adapters", "python_fastapi_lora.pt"),
            "java_springboot": os.path.join(current_dir, "adapters", "java_springboot_lora.pt"),
            "js_express":     os.path.join(current_dir, "adapters", "js_express_lora.pt"),
            "dotnet":         os.path.join(current_dir, "adapters", "dotnet_lora.pt"),
            "ai_ml":          os.path.join(current_dir, "adapters", "ai_ml_lora.pt"),
            "merger":         os.path.join(current_dir, "adapters", "merger_lora.pt"),
        }
        
        for name, path in adapter_files.items():
            self.manager.load_adapter(name, path)

        # Initialize domain-specific agents
        from agents.base_agent import BaseAgent

        # Define system prompts for the new frameworks
        FLUTTER_SYSTEM = (
            "You are an expert Flutter + Dart developer.\n"
            "Stack: Flutter SDK, Dart, Provider/Riverpod, GoRouter, HTTP/Dio.\n"
            "Always: write clean widgets, follow Dart style guide, write responsive layouts, and handle state management properly."
        )
        ANGULAR_SYSTEM = (
            "You are an expert Angular developer.\n"
            "Stack: TypeScript, Angular CLI, RxJS, NgRx, HttpClient.\n"
            "Always: use standalone components, implement proper lifecycle hooks, use reactive forms, and manage subscriptions."
        )
        NEXTJS_SYSTEM = (
            "You are an expert Next.js developer.\n"
            "Stack: React 18, Next.js (App Router), TypeScript, Tailwind CSS, Server Actions.\n"
            "Always: distinguish server and client components ('use client'), handle error.tsx boundaries, and optimize images/fonts."
        )
        FASTAPI_SYSTEM = (
            "You are an expert Python FastAPI developer.\n"
            "Stack: FastAPI, Pydantic v2, SQLAlchemy/SQLModel, Uvicorn, JWT Auth.\n"
            "Always: write async path operations, declare response_model schemas, use dependency injection, and write Pydantic validation."
        )
        SPRINGBOOT_SYSTEM = (
            "You are an expert Java Spring Boot developer.\n"
            "Stack: Java 17, Spring Boot 3, Spring Data JPA, Spring Security, Maven/Gradle.\n"
            "Always: use constructor injection, follow controller-service-repository pattern, handle exceptions globally with @ControllerAdvice, and secure endpoints."
        )
        EXPRESS_SYSTEM = (
            "You are an expert Node.js Express developer.\n"
            "Stack: JavaScript/TypeScript, Express, Knex/Prisma, JWT, Jest.\n"
            "Always: write modular routers, implement error handling middleware, validate requests using Joi/Zod, and follow RESTful path structures."
        )
        DOTNET_SYSTEM = (
            "You are an expert C# .NET Web API developer.\n"
            "Stack: .NET 8, Entity Framework Core, SQL Server, ASP.NET Core Identity.\n"
            "Always: write asynchronous controller methods (async/await), use dependency injection, follow the Repository/Unit of Work pattern, and structure projects with clean architecture principles."
        )

        self.agents = {
            Domain.REACT:          ReactAgent(self.model, self.manager, self.tokenizer),
            Domain.REACT_NATIVE:   ReactNativeAgent(self.model, self.manager, self.tokenizer),
            Domain.FLUTTER:        BaseAgent(self.model, self.manager, "flutter", FLUTTER_SYSTEM, self.tokenizer),
            Domain.ANGULAR:        BaseAgent(self.model, self.manager, "angular", ANGULAR_SYSTEM, self.tokenizer),
            Domain.NEXTJS:         BaseAgent(self.model, self.manager, "nextjs", NEXTJS_SYSTEM, self.tokenizer),
            Domain.PYTHON_BACKEND: PythonBackendAgent(self.model, self.manager, self.tokenizer),
            Domain.PYTHON_FASTAPI: BaseAgent(self.model, self.manager, "python_fastapi", FASTAPI_SYSTEM, self.tokenizer),
            Domain.JAVA_SPRINGBOOT: BaseAgent(self.model, self.manager, "java_springboot", SPRINGBOOT_SYSTEM, self.tokenizer),
            Domain.JS_EXPRESS:     BaseAgent(self.model, self.manager, "js_express", EXPRESS_SYSTEM, self.tokenizer),
            Domain.DOTNET:         BaseAgent(self.model, self.manager, "dotnet", DOTNET_SYSTEM, self.tokenizer),
            Domain.AI_ML:          AiMlAgent(self.model, self.manager, self.tokenizer),
            Domain.MERGER:         MergerAgent(self.model, self.manager, self.tokenizer),
        }
        self.orchestrator = Orchestrator()
        
        self.rag = LocalRAG(workspace_path=os.path.dirname(os.path.abspath(__file__)))
        self.rag.index_workspace()

    def run(self, task: str, frontend: str = None, backend: str = None, verbose: bool = True) -> dict:
        """End-to-end pipeline: task -> decompose -> agents -> merge -> output"""
        if verbose:
            print(f"\n[Orchestrator] Task: {task} | Frontend: {frontend} | Backend: {backend}")

        # If both stacks are explicitly defined, dynamically merge them into a single custom model for the project run!
        if frontend and backend:
            if verbose:
                print(f"[Pipeline] Merging frontend '{frontend}' and backend '{backend}' adapters into a single custom SLM...")
            self.manager.merge_adapters_to_base([frontend, backend])
            self.manager.is_merged = True
        else:
            self.manager.is_merged = False

        # Step 1: Decompose task into sub-tasks
        subtasks = self.orchestrator.decompose(task, frontend, backend)
        if verbose:
            for st in subtasks:
                print(f"  -> [{st.domain.value}] {st.instruction[:60]}...")

        # Step 2: Run non-merger tasks sequentially on the shared model
        agent_outputs = []
        merger_task = None

        for st in subtasks:
            if st.domain == Domain.MERGER:
                merger_task = st
                continue

            agent = self.agents.get(st.domain)
            if agent:
                if verbose:
                    print(f"\n[{st.domain.value.upper()} Agent] Swapping adapter & running...")
                
                # Retrieve relevant context from Parameterless Memory
                context = self.rag.retrieve_context(st.instruction)
                output = agent.run(st.instruction, context=context)
                agent_outputs.append(output)
                if verbose:
                    print(f"  Generated {len(output['code'])} characters of code.")

        # Step 3: Merge outputs if multiple sub-tasks were run
        final_output = None
        if merger_task and len(agent_outputs) > 1:
            if verbose:
                print(f"\n[Merger Agent] Swapping adapter & integrating {len(agent_outputs)} outputs...")
            merger: MergerAgent = self.agents[Domain.MERGER]
            final_output = merger.merge(agent_outputs, task)
        else:
            final_output = agent_outputs[0] if agent_outputs else {}

        return {
            "task": task,
            "sub_outputs": agent_outputs,
            "merged_output": final_output,
            "domains_used": [st.domain.value for st in subtasks],
        }

if __name__ == "__main__":
    # Test run
    pipeline = OrgSLMPipeline("checkpoints/base_pretrained.pt")
    
    # 1. Full-stack task
    result = pipeline.run(
        "Build a user authentication screen in React Native with JWT login, "
        "and the Django backend endpoint for login with token generation"
    )
    print("\n=== MERGED OUTPUT ===")
    print(result["merged_output"].get("code", ""))

    # 2. Single domain task
    result2 = pipeline.run("Write a Django serializer for a Product model with category and price")
    print("\n=== SINGLE DOMAIN OUTPUT ===")
    print(result2["merged_output"].get("code", ""))
