import json, re
from agents.base_agent import BaseAgent

class MergerAgent(BaseAgent):
    SYSTEM = """You are a full-stack integration specialist.
Given frontend (React/RN) and backend (Django REST) outputs:
1. Align API endpoint URLs between frontend fetch calls and backend URL patterns
2. Match TypeScript types to Django serializer field names
3. Generate axios/fetch client functions for each backend endpoint
4. Add JWT token interceptors to all API calls
5. Resolve any field name mismatches (snake_case ↔ camelCase)
6. Output: merged TypeScript API client + any backend fixes needed
Always output clean, production-ready integrated code."""

    def __init__(self, model, manager, tokenizer):
        super().__init__(model, manager, "merger", self.SYSTEM, tokenizer)

    def merge(self, agent_outputs: list, original_task: str) -> dict:
        """Consolidate all sub-agent outputs into integrated code"""

        # Build context block from all agent outputs
        context_parts = []
        for out in agent_outputs:
            context_parts.append(
                f"=== {out['agent'].upper()} OUTPUT ===\n{out['code']}"
            )
        context_str = "\n".join(context_parts)
        merger_instruction = f"""
Original task: {original_task}

Sub-agent outputs to integrate:
{context_str}

Tasks:
1. Identify all API endpoints in the Django code
2. Generate TypeScript fetch/axios functions for each endpoint
3. Ensure field names match between frontend and backend
4. Add proper error handling and loading states
5. Output the complete integrated API client layer
"""
        result = self.run(merger_instruction)
        result["type"] = "merged_fullstack"
        result["integration_report"] = self._build_report(agent_outputs, result)
        return result

    def _build_report(self, inputs: list, merged: dict) -> dict:
        """Build a summary of what was integrated"""
        report = {
            "agents_integrated": [o["agent"] for o in inputs],
            "endpoints_aligned": [],
            "type_conflicts_resolved": [],
        }

        # Extract endpoints from backend outputs
        for out in inputs:
            if out.get("type") == "django_backend":
                report["endpoints_aligned"].extend(out.get("endpoints", []))

        return report
