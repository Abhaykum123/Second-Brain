import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")
st.title("🧠 Second Brain")
st.caption("Groq-powered RAG personal knowledge assistant")

st.sidebar.header("Add Knowledge")
uploaded = st.sidebar.file_uploader("Upload document", type=["pdf", "txt", "md", "docx"])

if st.sidebar.button("Ingest File"):
    if uploaded is None:
        st.sidebar.warning("Please select a file first.")
    else:
        try:
            r = requests.post(
                f"{API_URL}/ingest/file",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=120,
            )
            if r.ok:
                st.sidebar.success(f"Added {r.json()['chunks']} chunks")
            else:
                st.sidebar.error(r.text)
        except Exception as e:
            st.sidebar.error(str(e))

st.sidebar.divider()
url = st.sidebar.text_input("Article URL")
if st.sidebar.button("Ingest URL"):
    if not url.strip():
        st.sidebar.warning("Enter a URL first.")
    else:
        try:
            r = requests.post(f"{API_URL}/ingest/url", json={"url": url}, timeout=120)
            if r.ok:
                st.sidebar.success(f"Added {r.json()['chunks']} chunks")
            else:
                st.sidebar.error(r.text)
        except Exception as e:
            st.sidebar.error(str(e))

st.subheader("Ask your Second Brain")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your knowledge...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your knowledge..."):
            try:
                r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
                answer = r.json()["answer"] if r.ok else f"Error: {r.text}"
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
