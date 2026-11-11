from agents.base_agent import BaseAgent

class ReactNativeAgent(BaseAgent):
    SYSTEM = """You are an expert React Native + Expo SDK 51 developer.
Stack: TypeScript, Expo Router v3, NativeWind, Zustand, MMKV, TanStack Query.
Always: handle safe area insets, use Platform.OS checks where needed,
write accessible components, handle keyboard avoiding, use proper navigation types."""

    def __init__(self, model, manager, tokenizer):
        super().__init__(model, manager, "react_native", self.SYSTEM, tokenizer)

    def run(self, instruction: str, context: dict = None) -> dict:
        result = super().run(instruction, context)
        result["type"] = "react_native_component"
        return result
