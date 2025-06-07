# main.py

import streamlit as st
from memory.semantic_memory import SemanticMemory
from agents.summarization_agent import SummarizationAgent
from agents.planning_agent import PlanningAgent

st.set_page_config(page_title="Memory-Augmented Multi-Agent Demo", layout="centered")
st.title("🧠 Multi-Agent Memory-Augmented LLM Demo")

# 1) Single shared semantic memory (FAISS-backed)
memory = SemanticMemory(embedding_model_name="all-MiniLM-L6-v2", dim=384, threshold=1.0)

# 2) Instantiate our agents, passing the same memory instance
summ_agent = SummarizationAgent(memory=memory, hf_model_name="facebook/opt-125m")
plan_agent = PlanningAgent(memory=memory, hf_model_name="facebook/opt-125m")

# 3) Sidebar: pick which agent to use
agent_choice = st.sidebar.radio("Choose Agent:", ["Summarization", "Planning"], index=0)

st.markdown("""
**How it works:**
1. Select “Summarization” or “Planning” in the sidebar.  
2. Enter your query and click **Submit**.  
3. The chosen agent:
   - **Cache Hit** → returns stored response.
   - **Cache Miss** → calls the LLM (via Hugging Face), returns generated text, and stores it in FAISS.
4. Future semantically similar queries may yield a **Cache Hit**.
""")

user_input = st.text_area("Enter your query:", height=120)

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Type a query before submitting.")
    else:
        if agent_choice == "Summarization":
            output = summ_agent.handle(user_input)
        else:
            output = plan_agent.handle(user_input)

        if output.startswith(f"[{agent_choice} Agent] ✅"):
            st.success(output)
        else:
            st.error(output)

