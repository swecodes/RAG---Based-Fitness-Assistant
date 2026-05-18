import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from rag import rag, search

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💪 Fitness RAG Assistant",
    page_icon="💪",
    layout="centered",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💪 Fitness Assistant")
    st.markdown(
        """
        **Powered by:**
        - 🦙 Ollama (Llama 3) — local LLM
        - 🔍 Minsearch — TF-IDF retrieval
        - 🐳 Docker — fully containerised

        **How it works:**
        1. Your question → search index
        2. Top-5 relevant exercises retrieved
        3. Context + question → Llama 3
        4. Grounded, hallucination-reduced answer

        ---
        *Built by Swetha Rathinavelu Saravanakumar*
        """
    )
    show_sources = st.toggle("Show retrieved sources", value=True)
    st.divider()
    st.caption("🟢 Connected to Ollama (llama3)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏋️ RAG-Based Fitness Assistant")
st.markdown(
    "Ask me anything about exercises — muscles targeted, equipment needed, "
    "how to perform a movement, or which activity type it falls under."
)

# ── Example questions ─────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
examples = [
    "How do I perform a deadlift safely?",
    "What muscles does a plank activate?",
    "What equipment do I need for a lat pulldown?",
    "Give me a cardio exercise I can do without equipment.",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question_input"] = ex

# ── Chat history ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question:",
    key="question_input",
    placeholder="e.g. What muscles does a push-up work?",
)

if st.button("Ask 💬", type="primary") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        result = rag(question.strip())

    st.session_state.history.append(result)

# ── Display history ───────────────────────────────────────────────────────────
for entry in reversed(st.session_state.history):
    st.divider()

    st.markdown(f"**🙋 You:** {entry['question']}")
    st.markdown(f"**🤖 Assistant:** {entry['answer']}")

    if show_sources and entry.get("source_docs"):
        with st.expander("📄 Retrieved Sources", expanded=False):
            for i, doc in enumerate(entry["source_docs"], 1):
                st.markdown(
                    f"**{i}. {doc.get('exercise_name', 'Unknown')}**  \n"
                    f"Type: `{doc.get('type_of_activity', '-')}` | "
                    f"Equipment: `{doc.get('type_of_equipment', '-')}` | "
                    f"Body Part: `{doc.get('body_part', '-')}`  \n"
                    f"*{doc.get('instructions', '')}*"
                )

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.info("Ask a question above to get started! 👆")
