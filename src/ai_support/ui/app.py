"""Streamlit chat interface backed only by the authenticated FastAPI API."""

import os

import streamlit as st

from ai_support.ui.client import APIClientError, SupportAPIClient

st.set_page_config(page_title="UOWD AI Support Assistant", page_icon="🎓")
st.title("UOWD AI Support Assistant")
st.caption("Synthetic Assessment 3 environment — do not enter real personal information.")

client = SupportAPIClient(os.getenv("AI_SUPPORT_API_URL", "http://127.0.0.1:8000"))
st.session_state.setdefault("access_token", None)
st.session_state.setdefault("audience", None)
st.session_state.setdefault("messages", [])

if not st.session_state.access_token:
    selected = st.selectbox("Demo audience", ("Student", "Staff"))
    default_id = "S100001" if selected == "Student" else "E100001"
    user_id = st.text_input("Synthetic user ID", value=default_id)
    login_code = st.text_input("Demo login code", type="password")
    if st.button("Sign in", type="primary"):
        try:
            login = client.login(user_id, login_code)
            st.session_state.access_token = login["access_token"]
            st.session_state.audience = login["audience"]
            st.rerun()
        except APIClientError as error:
            st.error(str(error))
    st.info("The server verifies the user ID and login code; this selector does not grant access.")
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.session_state.audience}**")
    if st.button("Sign out"):
        st.session_state.access_token = None
        st.session_state.audience = None
        st.session_state.messages = []
        st.rerun()

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.code(citation["citation"])
        if message.get("handoff_url"):
            st.link_button("Open official support page", message["handoff_url"])
        action = message.get("action")
        if action and not message.get("action_sent"):
            st.info(
                f"Draft to {action['department']} ({action['recipient']})\n\n"
                f"**{action['subject']}**\n\n{action['body']}"
            )
            if st.button("Confirm and send via Mailpit", key=f"confirm-{action['action_id']}"):
                try:
                    client.confirm(st.session_state.access_token, action["action_id"])
                    message["action_sent"] = True
                    st.success("The synthetic email was delivered to Mailpit.")
                    st.rerun()
                except APIClientError as error:
                    st.error(str(error))
        if message["role"] == "assistant" and message.get("message_id"):
            left, right = st.columns(2)
            if left.button("Helpful", key=f"helpful-{index}"):
                client.feedback(st.session_state.access_token, message["message_id"], True)
                st.toast("Feedback recorded")
            if right.button("Not helpful", key=f"not-helpful-{index}"):
                client.feedback(st.session_state.access_token, message["message_id"], False)
                st.toast("Feedback recorded")

if prompt := st.chat_input("Ask a UOWD support question", max_chars=3000):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        reply = client.chat(st.session_state.access_token, prompt)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply["answer"],
                "message_id": reply["message_id"],
                "citations": reply["citations"],
                "action": reply.get("action"),
                "handoff_url": reply.get("handoff_url"),
            }
        )
    except APIClientError as error:
        st.session_state.messages.append({"role": "assistant", "content": str(error)})
    st.rerun()
