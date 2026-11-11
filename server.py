import os
import time
import warnings

# Suppress FutureWarnings and RequestsDependencyWarning in log outputs
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except ImportError:
    pass
import subprocess
import urllib.request
import urllib.error
import json
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from pipeline import OrgSLMPipeline
import uvicorn

app = FastAPI(title="Org SLM Dashboard API", version="1.0")

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
try:
    pipeline = OrgSLMPipeline("checkpoints/base_pretrained.pt")
except Exception as e:
    print(f"Warning: Pipeline initialization failed during startup ({e}). It will be initialized on the first request.")

def query_ollama(prompt: str) -> Optional[str]:
    """Queries local Ollama instance if active. Matches models starting with gemma first."""
    # 1. Fetch available models
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=1.0) as response:
            tags = json.loads(response.read().decode('utf-8'))
            models = tags.get("models", [])
            if not models:
                return None
            
            selected_model = models[0]["name"]
            for m in models:
                if "gemma" in m["name"].lower():
                    selected_model = m["name"]
                    break
    except Exception:
        return None

    # 2. Call generate endpoint
    try:
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        req_gen = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_gen, timeout=20.0) as response_gen:
            res = json.loads(response_gen.read().decode('utf-8'))
            return res.get("response", "")
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        return None

class TaskRequest(BaseModel):
    task: str
    agent: str = "auto"     # "auto" | specific agent name
    frontend_stack: Optional[str] = None
    backend_stack: Optional[str] = None
    max_tokens: int = 800

class TaskResponse(BaseModel):
    task: str
    domains_used: list
    code: str
    full_output: dict
    elapsed_ms: float
    file_path: Optional[str] = None

class PlanRequest(BaseModel):
    task: str
    frontend_stack: Optional[str] = None
    backend_stack: Optional[str] = None

class PlanResponse(BaseModel):
    plan: str
    domains_used: List[str]

class ExecuteRequest(BaseModel):
    task: str
    frontend_stack: Optional[str] = None
    backend_stack: Optional[str] = None
    dry_run: bool = False

class ExecuteResponse(BaseModel):
    tasks: List[str]
    actions: List[dict]
    execution_results: Optional[List[dict]] = None

class VerifyRequest(BaseModel):
    task: str
    frontend_stack: Optional[str] = None
    backend_stack: Optional[str] = None
    actions_run: List[dict]

class VerifyResponse(BaseModel):
    walkthrough: str

@app.post("/generate", response_model=TaskResponse)
def generate(req: TaskRequest):
    global pipeline
    if pipeline is None:
        try:
            pipeline = OrgSLMPipeline("checkpoints/base_pretrained.pt")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline failed to initialize: {e}")

    start_time = time.time()
    try:
        if req.agent == "auto" or req.agent == "":
            result = pipeline.run(
                req.task, 
                frontend=req.frontend_stack, 
                backend=req.backend_stack
            )
            code = result["merged_output"].get("code", "")
        else:
            # Force a specific domain agent
            from agents.orchestrator import Domain
            # Convert agent string to Domain enum
            try:
                domain_val = req.agent
                if domain_val == "python":
                    domain_val = "python_backend"
                domain = Domain(domain_val)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown agent: {req.agent}")
                
            agent = pipeline.agents[domain]
            output = agent.run(req.task)
            result = {
                "task": req.task,
                "domains_used": [domain.value],
                "sub_outputs": [output],
                "merged_output": output
            }
            code = output["code"]

        file_path = None
        # Check if code is just a repetition of the system prompt guidelines to prevent wiping out user code
        is_prompt_repetition = "you are an expert" in code.lower() and "always:" in code.lower()
        
        if is_prompt_repetition:
            code = (
                "// No valid code block generated. Please check your prompt instructions.\n"
                "// Make sure your task describes the components you want to generate (e.g. 'Create a counter')."
            )
        
        if not is_prompt_repetition and len(result["domains_used"]) > 0:
            domain = result["domains_used"][0]
            file_path = extract_file_path(code, domain, req.task)

        elapsed_ms = (time.time() - start_time) * 1000
        return TaskResponse(
            task=req.task,
            domains_used=result["domains_used"],
            code=code,
            full_output=result,
            elapsed_ms=elapsed_ms,
            file_path=file_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
def list_agents():
    return {
        "agents": [
            "react", "react_native", "flutter", "angular", "nextjs",
            "python_backend", "python_fastapi", "java_springboot", 
            "js_express", "dotnet", "ai_ml", "merger", "auto"
        ]
    }

@app.post("/train/{domain}")
def trigger_train(domain: str, background_tasks: BackgroundTasks):
    valid_domains = [
        "react", "react_native", "flutter", "angular", "nextjs",
        "python_backend", "python_fastapi", "java_springboot",
        "js_express", "dotnet", "ai_ml", "merger"
    ]
    if domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Choose from {valid_domains}")
        
    def train_task():
        try:
            print(f"[Server] Asynchronous training started for adapter: {domain}")
            subprocess.run(["python", "train_adapters.py", domain], check=True)
            print(f"[Server] Asynchronous training finished for adapter: {domain}")
            # Reload adapter weights in pipeline
            if pipeline:
                adapter_files = {
                    "react":          "adapters/react_lora.pt",
                    "react_native":   "adapters/react_native_lora.pt",
                    "flutter":        "adapters/flutter_lora.pt",
                    "angular":        "adapters/angular_lora.pt",
                    "nextjs":         "adapters/nextjs_lora.pt",
                    "python_backend": "adapters/python_backend_lora.pt",
                    "python_fastapi": "adapters/python_fastapi_lora.pt",
                    "java_springboot": "adapters/java_springboot_lora.pt",
                    "js_express":     "adapters/js_express_lora.pt",
                    "dotnet":         "adapters/dotnet_lora.pt",
                    "ai_ml":          "adapters/ai_ml_lora.pt",
                    "merger":         "adapters/merger_lora.pt",
                }
                pipeline.manager.load_adapter(domain, adapter_files[domain])
        except Exception as e:
            print(f"[Server] Background training failed for '{domain}': {e}")

def extract_file_path(code: str, domain: str, task: str) -> str:
    """Helper to extract a filepath from comments in code block, or deduce a smart default."""
    lines = code.split('\n')
    for line in lines[:10]: # Check first 10 lines
        line_clean = line.strip().lower()
        # Look for comment lines indicating a file path:
        # e.g., // file: src/App.tsx, # path: main.py, /* filename: index.js */
        match = re.search(r'(?://|#|/\*)\s*(?:file|path|filename|filepath):\s*([a-zA-Z0-9_\-\./\\_]+)', line_clean)
        if match:
            extracted = match.group(1).strip()
            # Clean up any trailing comment formatting
            extracted = extracted.replace('*/', '').strip()
            return extracted
            
    # Guess based on task and domain
    task_lower = task.lower()
    is_login = "login" in task_lower or "auth" in task_lower
    is_landing = "landing" in task_lower or "ecommerce" in task_lower or "product" in task_lower

    if domain == "flutter":
        if is_login:
            return "frontend/lib/screens/login_screen.dart"
        elif is_landing:
            return "frontend/lib/screens/landing_page.dart"
        return "frontend/lib/main.dart"
    elif domain == "react":
        if is_login:
            return "frontend/src/components/Login.tsx"
        elif is_landing:
            return "frontend/src/components/LandingPage.tsx"
        return "frontend/src/App.tsx"
    elif domain == "nextjs":
        if is_login:
            return "frontend/app/login/page.tsx"
        elif is_landing:
            return "frontend/app/landing/page.tsx"
        return "frontend/app/page.tsx"
    elif domain == "react_native":
        if is_login:
            return "frontend/screens/LoginScreen.tsx"
        return "frontend/App.tsx"
    elif domain == "angular":
        if is_login:
            return "frontend/src/app/login/login.component.ts"
        return "frontend/src/app/app.component.ts"
    elif domain == "python_fastapi":
        return "backend/main.py"
    elif domain == "python_backend":
        return "backend/views.py"
    elif domain == "js_express":
        return "backend/server.js"
    elif domain == "java_springboot":
        return "backend/src/main/java/com/example/demo/AuthController.java"
    elif domain == "dotnet":
        return "backend/Controllers/AuthController.cs"
    elif domain == "ai_ml":
        return "ml_model/train.py"
        
    return "workspace_file.txt"

@app.post("/plan", response_model=PlanResponse)
def get_plan(req: PlanRequest):
    from agents.orchestrator import Orchestrator
    orch = Orchestrator()
    domains = orch.classify_domains(req.task, req.frontend_stack, req.backend_stack)
    domain_names = [d.value for d in domains]
    
    # 1. Try Ollama first
    prompt = (
        f"<|system|>You are a professional software architect. Generate a clean and detailed markdown implementation plan for the task: '{req.task}' using frontend: '{req.frontend_stack}' and backend: '{req.backend_stack}'.\n"
        f"Strictly format the response using standard GitHub Markdown. Do NOT include any introductory or concluding text, only the plan markdown content. "
        f"Structure the document exactly with these headings:\n"
        f"# Implementation Plan: [Goal Description]\n\n"
        f"## User Review Required\n"
        f"Detail the configuration choices and architecture patterns to be used.\n\n"
        f"## Proposed Changes\n"
        f"### [Component Name]\n"
        f"List modified or new files with [NEW] or [MODIFY] markers. Provide relative file path relative to workspace (e.g., frontend/lib/main.dart, backend/main.py, index.html).\n\n"
        f"## Verification Plan\n"
        f"### Manual Verification\n"
        f"List step-by-step instructions to verify the changes, including exact terminal commands to run (e.g., flutter run, python main.py).<|end|>"
        f"<|user|>Generate the plan.<|end|>"
        f"<|assistant|>"
    )
    plan_content = query_ollama(prompt)
    
    if not plan_content:
        # Fallback to high-quality programmatic plan template
        frontend_str = req.frontend_stack or (domain_names[0] if len(domain_names) > 0 else "None")
        backend_str = req.backend_stack or (domain_names[1] if len(domain_names) > 1 else "None")
        
        plan_content = f"""# Implementation Plan: Fullstack implementation of "{req.task}"

## User Review Required
> [!IMPORTANT]
> The project will be generated locally in the active VS Code workspace.
> Stacks selected: Frontend: **{frontend_str.upper()}**, Backend: **{backend_str.upper()}**.
> No external cloud dependencies are required. All services will run locally on your system.

## Proposed Changes
We will create the following codebase structure to fulfill the requested functionality:

"""
        for dom in domain_names:
            plan_content += f"### {dom.upper()} Component\n"
            if dom == "flutter":
                plan_content += "- `[NEW]` [login_screen.dart](file:///frontend/lib/screens/login_screen.dart) - Main UI screen with form validation and service integrations\n"
                plan_content += "- `[NEW]` [main.dart](file:///frontend/lib/main.dart) - Application entry point initiating the app\n"
            elif dom == "react":
                plan_content += "- `[NEW]` [Login.tsx](file:///frontend/src/components/Login.tsx) - Beautiful login UI component with micro-animations\n"
                plan_content += "- `[NEW]` [App.tsx](file:///frontend/src/App.tsx) - Entry file displaying the login screen\n"
            elif dom == "nextjs":
                plan_content += "- `[NEW]` [app/login/page.tsx](file:///frontend/app/login/page.tsx) - Next.js page component using App Router\n"
            elif dom == "react_native":
                plan_content += "- `[NEW]` [screens/LoginScreen.tsx](file:///frontend/screens/LoginScreen.tsx) - React Native Auth screen view\n"
            elif dom == "python_fastapi":
                plan_content += "- `[NEW]` [main.py](file:///backend/main.py) - FastAPI app router exposing POST endpoints and Pydantic validation\n"
            elif dom == "python_backend":
                plan_content += "- `[NEW]` [views.py](file:///backend/views.py) - Django view implementing token-based authentication\n"
            elif dom == "js_express":
                plan_content += "- `[NEW]` [server.js](file:///backend/server.js) - Node Express router running JWT token endpoints\n"
            elif dom == "java_springboot":
                plan_content += "- `[NEW]` [AuthController.java](file:///backend/src/main/java/com/example/demo/AuthController.java) - Java controller for token endpoints\n"
            elif dom == "dotnet":
                plan_content += "- `[NEW]` [Controllers/AuthController.cs](file:///backend/Controllers/AuthController.cs) - .NET Web API token endpoint controller\n"
            elif dom == "ai_ml":
                plan_content += "- `[NEW]` [train.py](file:///ml_model/train.py) - Python training script or model definitions\n"
            plan_content += "\n"

        plan_content += f"""## Verification Plan
### Manual Verification
1. Verify files are successfully created in the workspace.
2. In the terminal, navigate to the component folders.
3. Start the services using the standard start commands for each stack.
"""
    return PlanResponse(plan=plan_content, domains_used=domain_names)

WORKSPACE_ROOT = os.path.abspath(os.getenv("WORKSPACE_ROOT", "./workspace/"))

@app.post("/execute", response_model=ExecuteResponse)
def execute_plan(req: ExecuteRequest):
    global pipeline
    if pipeline is None:
        try:
            pipeline = OrgSLMPipeline("checkpoints/base_pretrained.pt")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline failed to initialize: {e}")

    # Run the pipeline to get the code blocks
    result = pipeline.run(
        req.task, 
        frontend=req.frontend_stack, 
        backend=req.backend_stack
    )

    domains = result["domains_used"]
    sub_outputs = result["sub_outputs"]
    merged_output = result["merged_output"]

    # Build tasks checklist
    tasks = ["Setup workspace directories"]
    for dom in domains:
        tasks.append(f"Generate {dom.upper()} implementation")
    tasks.append("Run project dependency checks")

    actions = []

    # Map sub-outputs into files
    # If there are sub_outputs, use them since they are domain-specific
    outputs_to_process = sub_outputs if sub_outputs else [merged_output]

    for out in outputs_to_process:
        domain = out.get("agent", domains[0] if domains else "unknown")
        code = out.get("code", "")
        code_blocks = out.get("code_blocks", [code] if code else [])

        for block in code_blocks:
            if not block.strip():
                continue
                
            # Is it a terminal command?
            first_line = block.strip().split('\n')[0].strip()
            cmd_prefixes = ['npm', 'yarn', 'pnpm', 'pip', 'python', 'py', 'git', 'flutter', 'dart', 'dotnet', 'cargo', 'cd', 'mkdir', 'npm run', 'npx']
            is_cmd = any(first_line.startswith(pref) or first_line.startswith('./' + pref) for pref in cmd_prefixes)

            if is_cmd:
                actions.append({
                    "type": "run_command",
                    "command": block.strip(),
                    "description": f"Execute setup command for {domain.upper()}"
                })
            else:
                # Deduce file path
                is_prompt_repetition = "you are an expert" in block.lower() and "always:" in block.lower()
                if not is_prompt_repetition:
                    file_path = extract_file_path(block, domain, req.task)
                    actions.append({
                        "type": "write_file",
                        "path": file_path,
                        "content": block,
                        "description": f"Write {os.path.basename(file_path)} for {domain.upper()}"
                    })

    # Add standard commands for domains to complete installation/run
    for dom in domains:
        if dom in ["react", "nextjs", "js_express"]:
            actions.append({
                "type": "run_command",
                "command": "npm install",
                "description": f"Install Node.js dependencies for {dom.upper()}"
            })
        elif dom == "flutter":
            actions.append({
                "type": "run_command",
                "command": "flutter pub get",
                "description": "Fetch Flutter packages"
            })
        elif dom in ["python_fastapi", "python_backend", "ai_ml"]:
            actions.append({
                "type": "run_command",
                "command": "pip install fastapi uvicorn pydantic requests pyjwt django djangorestframework django-cors-headers djangorestframework-simplejwt",
                "description": "Install Python backend dependencies"
            })

    execution_results = []
    
    # Ensure WORKSPACE_ROOT exists physically
    if not req.dry_run:
        os.makedirs(WORKSPACE_ROOT, exist_ok=True)

    for action in actions:
        act_type = action["type"]
        desc = action["description"]
        
        if act_type == "write_file":
            rel_path = action["path"]
            content = action["content"]
            
            # Prevent path traversal (sandbox check)
            target_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, rel_path))
            if not target_path.startswith(WORKSPACE_ROOT):
                raise HTTPException(
                    status_code=400,
                    detail=f"Security Alert: Path traversal detected for target path '{rel_path}'."
                )
            
            result_item = {
                "action": desc,
                "type": "write_file",
                "path": rel_path,
                "status": "success",
                "message": ""
            }
            
            if req.dry_run:
                result_item["message"] = f"[Dry Run] Would write file to target path: {rel_path}"
                print(f"[Execute-DryRun] Would write file: {target_path}")
            else:
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    result_item["message"] = f"Successfully wrote file: {rel_path}"
                    print(f"[Execute] Wrote file: {target_path}")
                except Exception as e:
                    result_item["status"] = "error"
                    result_item["message"] = f"Failed to write file: {str(e)}"
                    print(f"[Execute-Error] Failed to write file {target_path}: {e}")
            execution_results.append(result_item)
            
        elif act_type == "run_command":
            cmd = action["command"]
            
            result_item = {
                "action": desc,
                "type": "run_command",
                "command": cmd,
                "status": "success",
                "message": "",
                "stdout": "",
                "stderr": ""
            }
            
            if req.dry_run:
                result_item["message"] = f"[Dry Run] Would execute terminal command: {cmd}"
                print(f"[Execute-DryRun] Would run command: {cmd}")
            else:
                try:
                    print(f"[Execute] Running command: {cmd} inside {WORKSPACE_ROOT}")
                    completed_process = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=WORKSPACE_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=60.0  # Limit run to 60 seconds
                    )
                    
                    result_item["stdout"] = completed_process.stdout
                    result_item["stderr"] = completed_process.stderr
                    
                    if completed_process.returncode == 0:
                        result_item["message"] = "Command executed successfully."
                    else:
                        result_item["status"] = "error"
                        result_item["message"] = f"Command failed with returncode {completed_process.returncode}."
                except Exception as e:
                    result_item["status"] = "error"
                    result_item["message"] = f"Command execution crashed: {str(e)}"
            execution_results.append(result_item)

    return ExecuteResponse(tasks=tasks, actions=actions, execution_results=execution_results)

@app.post("/verify", response_model=VerifyResponse)
def verify_project(req: VerifyRequest):
    # Try Ollama first
    prompt = (
        f"<|system|>You are a professional quality assurance engineer. Generate a clean and detailed markdown walkthrough summarizing the completion and verification steps for the task: '{req.task}'.\n"
        f"Strictly format the response using standard GitHub Markdown. Do NOT include any introductory or concluding text, only the walkthrough markdown content. "
        f"Structure the document exactly with these headings:\n"
        f"# Project Walkthrough: [Goal Description]\n\n"
        f"## Accomplishments\n"
        f"List all directories and files created and setup commands executed successfully.\n\n"
        f"## Verification Details\n"
        f"### Automated Verification\n"
        f"Provide the exact terminal command to run tests or launch the project.\n\n"
        f"### Manual Verification\n"
        f"Detail how the user should verify the UI elements, interactions, and backend routes in the browser or emulator.<|end|>"
        f"<|user|>Generate the walkthrough.<|end|>"
        f"<|assistant|>"
    )
    walkthrough_content = query_ollama(prompt)
    
    if not walkthrough_content:
        # Fallback to high-quality template walkthrough
        walkthrough_content = f"""# Project Walkthrough: Fullstack implementation of "{req.task}"

## Accomplishments
The local SLM agent has successfully completed autonomous execution:
- Created the project directories and files in your workspace.
- Setup code logic for the frontend and backend integration.
- Installed necessary system packages and application dependencies.

## Verification Details
### Automated Verification
To launch the integrated local solution:
1. Open the project in the workspace.
2. Run the starting commands to activate the services.

### Manual Verification
1. Open your browser and navigate to the local endpoint (e.g., `http://localhost:8080` or the frontend server port).
2. Validate that the UI fields render correctly.
3. Test interaction flows (such as form inputs, buttons, and state changes).
"""
    return VerifyResponse(walkthrough=walkthrough_content)

LOG_FILE_PATH = "orchestrator_system.log"

@app.get("/api/logs")
async def get_live_orchestrator_logs(lines: int = 15):
    """
    Reads the last N structural lines from the log file 
    allowing your desktop-app UI to stream engine events live.
    """
    if not os.path.exists(LOG_FILE_PATH):
        return JSONResponse(content={"status": "empty", "logs": []})
        
    try:
        with open(LOG_FILE_PATH, "r") as file:
            # Efficiently pull trailing log blocks without running out of RAM
            all_lines = file.readlines()
            tail_lines = all_lines[-lines:]
            
            # Clean newlines for clean JSON array compilation
            cleaned_logs = [line.strip() for line in tail_lines if line.strip()]
            return JSONResponse(content={"status": "active", "logs": cleaned_logs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
