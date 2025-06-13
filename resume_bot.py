import streamlit as st
from invoke_llm_api import qa_chain

# --- Sidebar ---s
with st.sidebar:
    st.title("🦙💬 Alfred")
    st.caption("🚀 A Chatbot for Karan's Professional Insights")
    st.markdown("📝 [View Resume](https://drive.google.com/file/d/1aesz8fhFe4yximSbJkKtcaSfbTTpWQOH/view)")
    st.markdown("👾 [GitHub Profile](https://github.com/karanpodar)")
    st.button("🧹 Clear Chat History", on_click=lambda: st.session_state.update({
        "messages": [{"role": "assistant", "content": "Hello Master Wayne! How may I assist you today?"}]
    }))

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello Master Wayne! How can I help you?"}]

# --- Render Chat History ---
def render_message(msg):
    st.chat_message(msg["role"]).write(msg["content"])

for msg in st.session_state.messages:
    render_message(msg)

# --- Handle User Input ---
if user_input := st.chat_input("Ask me anything about Karan..."):
    user_msg = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # --- Generate Assistant Response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = qa_chain(user_input)
            placeholder = st.empty()
            full_response = ""
            for chunk in response_stream:
                full_response += chunk
                placeholder.markdown(full_response)
        assistant_msg = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(assistant_msg)
