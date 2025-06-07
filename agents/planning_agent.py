# agents/planning_agent.py

from agents.base_agent import BaseAgent

class PlanningAgent(BaseAgent):
    def __init__(self, memory, hf_model_name="facebook/opt-125m"):
        super().__init__(
            name="Planning",
            memory=memory,
            hf_model_name=hf_model_name,
            llm_max_length=64,
            llm_temperature=0.7,
        )

    def _build_prompt(self, user_input: str) -> str:
        # Prepend planning instruction
        return f"You are an AI assistant that helps plan tasks. Given:\n\n{user_input}\n\nProvide a step-by-step plan:"
