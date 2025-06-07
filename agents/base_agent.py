# agents/base_agent.py

from memory.semantic_memory import SemanticMemory
from llm.hf_llm import generate_with_hf

class BaseAgent:
    """
    Base agent logic:
    1) Check semantic memory for a similar input.
    2) If hit → return cached response.
    3) If miss → build prompt, call LLM, store new (input→response), return generated.
    """

    def __init__(
        self,
        name: str,
        memory: SemanticMemory,
        hf_model_name: str = "gpt2",              # ← switched to gpt2
        llm_max_length: int = 64,
        llm_temperature: float = 0.7,
    ):
        self.name = name
        self.memory = memory
        self.model_name = hf_model_name
        self.max_length = llm_max_length
        self.temperature = llm_temperature

    def _build_prompt(self, user_input: str) -> str:
        # By default, pass the input unchanged; subclasses override this.
        return user_input

    def handle(self, user_input: str) -> str:
        # 1) Query semantic memory
        hit, cached_response = self.memory.query(user_input)
        if hit:
            return f"[{self.name} Agent] ✅ Cache Hit:\n\n{cached_response}"

        # 2) Cache miss → build prompt & call LLM
        prompt = self._build_prompt(user_input)
        try:
            generated = generate_with_hf(
                model_name=self.model_name,
                prompt=prompt,
                max_length=self.max_length,
                temperature=self.temperature,
            )
        except Exception as e:
            # Surface the error message instead of crashing
            return f"[{self.name} Agent] ⚠️ LLM Error:\n\n{e}"

        # 3) Store new (input→generated) in memory
        self.memory.add_to_memory(user_input, generated)
        return f"[{self.name} Agent] ❌ Cache Miss → LLM Response:\n\n{generated}"


