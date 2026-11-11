import re
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class Domain(Enum):
    REACT = "react"
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    ANGULAR = "angular"
    NEXTJS = "nextjs"
    PYTHON_BACKEND = "python_backend"  # Django
    PYTHON_FASTAPI = "python_fastapi"
    JAVA_SPRINGBOOT = "java_springboot"
    JS_EXPRESS = "js_express"
    DOTNET = "dotnet"
    AI_ML = "ai_ml"
    MERGER = "merger"
    ORCHESTRATOR = "orchestrator"

@dataclass
class SubTask:
    domain: Domain
    instruction: str
    context: dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list) # task IDs
    task_id: str = ""

class Orchestrator:
    """
    Classifies developer intent, decomposes into sub-tasks,
    routes each to the right sub-agent, collects outputs,
    then sends to the merger.
    """
    # Fast regex patterns for domain routing
    DOMAIN_PATTERNS = {
        Domain.REACT: [
            r"\breact\b(?![\s-]native)", r"\bjsx\b", r"\btsx\b",
            r"\bcomponent\b", r"\bhook\b", r"\buseState\b", r"\bzustand\b",
            r"\btailwind\b", r"\bvite\b",
        ],
        Domain.REACT_NATIVE: [
            r"\breact native\b", r"\bexpo\b", r"\bnativewind\b",
            r"\bflatlist\b", r"\bnavigation\b", r"\bmobile\b",
            r"\bios\b", r"\bandroid\b",
        ],
        Domain.FLUTTER: [
            r"\bflutter\b", r"\bdart\b", r"\bwidget\b", r"\bstatefulwidget\b",
            r"\bstatelesswidget\b", r"\bflutter\b.*\bapp\b",
        ],
        Domain.ANGULAR: [
            r"\bangular\b", r"\bngmodule\b", r"\bcomponent\b.*\bangular\b",
        ],
        Domain.NEXTJS: [
            r"\bnext\.?js\b", r"\bserver component\b", r"\bgetstaticprops\b",
        ],
        Domain.PYTHON_BACKEND: [
            r"\bdjango\b", r"\bdrf\b", r"\bviewset\b", r"\bserializer\b",
            r"\bcelery\b", r"\bpostgres\b",
        ],
        Domain.PYTHON_FASTAPI: [
            r"\bfastapi\b", r"\buvicorn\b", r"\bapirouter\b", r"\bpydantic\b",
        ],
        Domain.JAVA_SPRINGBOOT: [
            r"\bspring\b", r"\bspringboot\b", r"\bjava\b", r"\bautowired\b", r"\bcontroller\b",
        ],
        Domain.JS_EXPRESS: [
            r"\bexpress\b", r"\bnode\b", r"\bexpress\.?js\b", r"\brouter\b.*\bexpress\b",
        ],
        Domain.DOTNET: [
            r"\b\.net\b", r"\bcsharp\b", r"\bc#\b", r"\bwebapi\b", r"\bcontroller\b.*\b\.net\b",
        ],
        Domain.AI_ML: [
            r"\bpytorch\b", r"\bhuggingface\b", r"\brag\b", r"\bllm\b",
            r"\bslm\b", r"\blangchain\b", r"\blora\b", r"\bfine.?tun\b",
            r"\bembedding\b", r"\bvector\b", r"\bagent\b.*\bai\b",
            r"\bml\b", r"\bdeep learning\b", r"\bneural\b",
        ],
    }

    def classify_domains(self, task: str, frontend: str = None, backend: str = None) -> List[Domain]:
        """Detect which domains a task touches, or use explicit overrides"""
        matched = []
        
        # 1. Check explicit overrides first
        if frontend:
            try:
                matched.append(Domain(frontend))
            except ValueError:
                pass
        if backend:
            try:
                matched.append(Domain(backend))
            except ValueError:
                pass
                
        # 2. If no overrides, run regex matching
        if not matched:
            task_lower = task.lower()
            for domain, patterns in self.DOMAIN_PATTERNS.items():
                if any(re.search(p, task_lower) for p in patterns):
                    matched.append(domain)
                    
        # Default to react/python backend if no pattern is matched, but let's default to python_backend
        return matched if matched else [Domain.PYTHON_BACKEND]

    def decompose(self, task: str, frontend: str = None, backend: str = None) -> List[SubTask]:
        """Break a full-stack task into parallel sub-tasks"""
        domains = self.classify_domains(task, frontend, backend)
        subtasks = []

        for i, domain in enumerate(domains):
            instruction = self._specialize_instruction(task, domain)
            st = SubTask(
                domain=domain,
                instruction=instruction,
                task_id=f"task_{i}_{domain.value}",
            )
            subtasks.append(st)

        # If multiple domains are touched, add a merger task
        if len(domains) > 1:
            merger_ids = [st.task_id for st in subtasks]
            subtasks.append(SubTask(
                domain=Domain.MERGER,
                instruction=f"Integrate all outputs for: {task}",
                depends_on=merger_ids,
                task_id="task_merger",
            ))

        return subtasks

    def _specialize_instruction(self, task: str, domain: Domain) -> str:
        prefixes = {
            Domain.REACT: "Write the React/TypeScript frontend code for: ",
            Domain.REACT_NATIVE: "Write the React Native/Expo mobile code for: ",
            Domain.FLUTTER: "Write the Flutter/Dart mobile/desktop code for: ",
            Domain.ANGULAR: "Write the Angular/TypeScript code for: ",
            Domain.NEXTJS: "Write the Next.js/React code for: ",
            Domain.PYTHON_BACKEND: "Write the Django REST Framework backend code for: ",
            Domain.PYTHON_FASTAPI: "Write the Python FastAPI backend code for: ",
            Domain.JAVA_SPRINGBOOT: "Write the Java Spring Boot backend code for: ",
            Domain.JS_EXPRESS: "Write the Node.js Express backend code for: ",
            Domain.DOTNET: "Write the C# .NET Web API backend code for: ",
            Domain.AI_ML: "Write the Python AI/ML implementation for: ",
        }
        return prefixes.get(domain, "") + task
