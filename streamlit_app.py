import streamlit as st
import requests

API_URL = "https://sagar-kc7--nepal-law-assistant-nepallawapi-ask.modal.run"

st.set_page_config(
    page_title="Nepal Law Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Nepal Law Assistant")
st.markdown("Ask any question about **Nepali laws** in plain English.")
st.markdown("---")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
if question := st.chat_input("Ask a legal question..."):
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # call API
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )
                answer = response.json().get("answer", "Sorry, could not get an answer.")
            except Exception as e:
                answer = f"Error connecting to API: {str(e)}"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("Powered by Llama 3.2 + RAG | Data: Constitution of Nepal 2015")
