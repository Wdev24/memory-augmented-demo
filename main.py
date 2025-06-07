import streamlit as st
from memory.semantic_memory import SemanticMemory
from agents.summarization_agent import SummarizationAgent
from agents.planning_agent import PlanningAgent

st.set_page_config(page_title="Memory-Augmented Multi-Agent Demo", layout="centered")
st.title("🧠 Multi-Agent Memory-Augmented LLM Demo")

# 1) Shared semantic memory (cosine-based)
memory = SemanticMemory(embedding_model_name="all-MiniLM-L6-v2", threshold=0.7)

# 2) Instantiate agents with gpt2
summ_agent = SummarizationAgent(memory=memory, hf_model_name="gpt2")
plan_agent = PlanningAgent(memory=memory, hf_model_name="gpt2")

# 3) Sidebar selector
agent_choice = st.sidebar.radio("Choose Agent:", ["Summarization", "Planning"], index=0)

st.markdown("""
**How it works:**
1. Select “Summarization” or “Planning.”  
2. Enter your query and click **Submit**.  
3. The chosen agent:
   - **Cache Hit** → returns stored response.
   - **Cache Miss** → calls the LLM, returns generated text, and caches it.
4. Similar queries may yield **Cache Hits**.
""")

user_input = st.text_area("Enter your query:", height=120)

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Type a query before submitting.")
    else:
        output = summ_agent.handle(user_input) if agent_choice == "Summarization" else plan_agent.handle(user_input)
        if output.startswith(f"[{agent_choice} Agent] ✅"):
            st.success(output)
        else:
            st.error(output)
