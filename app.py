import streamlit as st
import os
from rag_engine import load_and_index_pdf, get_qa_chain

st.set_page_config(
    page_title="Pakistan Laws Chatbot",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Pakistan Laws Chatbot")
st.markdown("Pakistan ke labor laws ke baare mein koi bhi sawaal poochein!")
st.divider()

# Sidebar mein token lo
with st.sidebar:
    st.header("🔑 Setup")
    hf_token = st.text_input(
        "HuggingFace Token daalo:",
        type="password",
        placeholder="hf_xxxxxxxxxx",
        help="https://huggingface.co/settings/tokens se lo"
    )
    st.markdown("👆 Token [yahan se lo](https://huggingface.co/settings/tokens)")

if not hf_token:
    st.info("👈 Pehle sidebar mein HuggingFace token daalo!")
    st.stop()

@st.cache_resource
def initialize(token):
    vectorstore, embeddings = load_and_index_pdf("data/labor_law.pdf", token)
    answer = get_qa_chain(vectorstore, token)
    return answer

with st.spinner("Laws database load ho raha hai..."):
    answer_fn = initialize(hf_token)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Sawaal likhein... jaise: overtime ka kya rule hai?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Jawab dhundh raha hai..."):
            result = answer_fn(question)
            st.write(result)

    st.session_state.messages.append({"role": "assistant", "content": result})