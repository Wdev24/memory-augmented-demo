# agents/summarization_agent.py

from agents.base_agent import BaseAgent

class SummarizationAgent(BaseAgent):
    def __init__(self, memory, hf_model_name="gpt2"):
        super().__init__(
            name="Summarization",
            memory=memory,
            hf_model_name=hf_model_name,
            llm_max_length=64,
            llm_temperature=0.7,
        )

    def _build_prompt(self, user_input: str) -> str:
        # Prepend summarization instruction
        return f"Please summarize the following content:\n\n{user_input}\n\nSummary:"

