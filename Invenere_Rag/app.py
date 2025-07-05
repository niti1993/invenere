import streamlit as st
from rag_engine.rag_pipeline import RAGPipeline

st.title("Invenere: Smart Enterprise Search")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question:")

if st.button("Search") and query:
    # Use conversation history if desired
    answer, sources = pipeline.query(query, return_sources=True, history=st.session_state.history)

    st.session_state.history.append((query, answer, sources))

    st.write("### 🧠 LLaMA's Response")
    st.write(answer)

    st.write("### 📄 Source files used")
    for src in sources:
        if isinstance(src, dict):
            st.markdown(f"- `{src.get('source', 'Unknown')}`")
        else:
            st.markdown(f"- `{src}`")

if st.session_state.history:
    st.write("## Previous Q&A")
    for q, a, s in st.session_state.history[::-1]:
        st.markdown(f"**Q:** {q}\n\n**A:** {a}")
        st.markdown("**Sources:**")
        for src in s:
            if isinstance(src, dict):
                st.markdown(f"- `{src.get('source', 'Unknown')}`")
            else:
                st.markdown(f"- `{src}`")
