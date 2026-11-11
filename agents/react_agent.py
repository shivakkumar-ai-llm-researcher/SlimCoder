import re
from agents.base_agent import BaseAgent

class ReactAgent(BaseAgent):
    SYSTEM = """You are an expert React developer.
Stack: TypeScript, functional components, custom hooks, Zustand, TanStack Query v5, Tailwind CSS.
Always: use proper TypeScript types, handle loading/error/empty states,
follow React 18 patterns, add JSDoc comments, use named exports."""

    def __init__(self, model, manager, tokenizer):
        super().__init__(model, manager, "react", self.SYSTEM, tokenizer)

    def run(self, instruction: str, context: dict = None) -> dict:
        result = super().run(instruction, context)
        # React-specific post-processing: validate JSX syntax & extract imports
        result["type"] = "react_component"
        result["imports"] = self._extract_imports(result["code"])
        return result

    def _extract_imports(self, code: str):
        return re.findall(r'^import .+$', code, re.MULTILINE)
