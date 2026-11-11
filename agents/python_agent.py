import re
from agents.base_agent import BaseAgent

class PythonBackendAgent(BaseAgent):
    SYSTEM = """You are an expert Django REST Framework developer.
Stack: Django 4.2, DRF 3.14, PostgreSQL, JWT (djangorestframework-simplejwt), Celery, Redis.
Always: use class-based views, add permission classes, write serializer validation,
handle pagination, add docstrings, write unit test stubs."""

    def __init__(self, model, manager, tokenizer):
        super().__init__(model, manager, "python_backend", self.SYSTEM, tokenizer)

    def run(self, instruction: str, context: dict = None) -> dict:
        import ast
        
        max_retries = 2
        result = None
        for attempt in range(max_retries + 1):
            result = super().run(instruction, context)
            code = result.get("code", "")
            
            clean_code = code
            if clean_code.startswith("```"):
                lines = clean_code.split('\n')
                if len(lines) > 2:
                    clean_code = '\n'.join(lines[1:-1])
                    
            try:
                ast.parse(clean_code)
                break
            except SyntaxError as e:
                if attempt < max_retries:
                    print(f"[AST Parser] Caught SyntaxError: {e}. Retrying ({attempt+1}/{max_retries})...")
                    instruction += f"\n\nYou made a syntax error: {e}. Please fix it."
                else:
                    print(f"[AST Parser] Max retries reached. Returning code with syntax errors.")
                    
        result["type"] = "django_backend"
        result["endpoints"] = self._extract_endpoints(result["code"])
        return result

    def _extract_endpoints(self, code: str):
        return re.findall(r'path\(["\']([^"\']+)["\']', code)
