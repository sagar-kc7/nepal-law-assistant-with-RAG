import streamlit as st
import requests
import re

API_URL = "http://localhost:8000"  # TODO: update to your Modal URL once redeployed

def sanitize_markdown(text: str) -> str:
    # Escape leading "N." patterns so markdown doesn't treat them as list items
    return re.sub(r"(?m)^(\d+)\.", r"\1\\.", text)

st.set_page_config(
    page_title="Nepal Law Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Nepal Law Assistant")
st.markdown("Ask any question about the **Constitution of Nepal** in plain English.")
st.markdown("---")

ROUTE_LABELS = {
    "specific_lookup": "🔍 Direct lookup (hybrid retrieval + reranking)",
    "broad_summary": "🌳 Broad summary (RAPTOR hierarchical retrieval)",
    "no_retrieval_needed": "💬 No retrieval needed",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"{ROUTE_LABELS.get(msg.get('route'), 'Sources')}"):
                for src in msg["sources"]:
                    st.markdown(src)

if question := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            try:
                ask_response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
                answer = ask_response.json().get("answer", "Sorry, could not get an answer.")

                search_response = requests.post(f"{API_URL}/search", json={"question": question}, timeout=60)
                search_data = search_response.json()
                route = search_data.get("route", "specific_lookup")

                sources = []
                for chunk in search_data.get("chunks", []):
                    if route == "broad_summary":
                        content = sanitize_markdown(chunk.get('content', ''))
                        sources.append(
                            f"**Cluster {chunk.get('cluster_id')}** — Articles: {chunk.get('child_articles', '')[:80]}...\n\n{content}"
                        )
                    else:
                        content = sanitize_markdown(chunk.get('content', ''))
                        sources.append(
                            f"**Article {chunk.get('article')}** (Part {chunk.get('part')})\n\n{content}"
                        )
            except Exception as e:
                answer = f"Error connecting to API: {str(e)}"
                route = None
                sources = []

        st.write(answer)
        if sources:
            with st.expander(f"{ROUTE_LABELS.get(route, 'Sources')}"):
                for src in sources:
                    st.markdown(src)
                    st.markdown("---")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "sources": sources,
        })

st.markdown("---")
st.caption("Powered by Llama 3.3 70B (Groq) + Hybrid Retrieval + Contextual Chunking + Reranking + RAPTOR | Data: Constitution of Nepal 2015")