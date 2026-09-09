"""Sign-in / sign-up screen."""

import streamlit as st

from config import APP_TAGLINE, APP_TITLE
from core import auth
from ui import state, theme


def render() -> None:
    theme.hero(APP_TITLE, APP_TAGLINE)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("#### What this project does")
        st.markdown(
            """
Most chatbots answer from the public internet, so they know nothing about
**your** notes. This assistant reads the material you upload and answers only
from it.

The same question is answered two different ways so the methods can be compared:

* **Approach 1 - Modern.** Notes are chunked, embedded into vectors and stored
  in a FAISS vector store. Relevant chunks are retrieved and sent to a language
  model through OpenRouter, which writes the answer. It also keeps a memory of
  the conversation.
* **Approach 2 - Traditional.** The same chunks are scored with **TF-IDF** and
  **cosine similarity**. The closest passage becomes the answer. No external
  service, no cost.
            """
        )
        st.info(
            "New here? Open the **Create account** tab, register with any email "
            "and password, then sign in.",
            icon="ℹ️",
        )

    with right:
        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@college.edu")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Sign in", type="primary", width="stretch"
                )

            if submitted:
                ok, message, record = auth.login(email, password)
                if ok:
                    state.sign_in(record)
                    st.rerun()
                else:
                    st.error(message)

        with sign_up_tab:
            with st.form("register_form"):
                name = st.text_input("Full name")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input(
                    "Password", type="password", key="reg_pw",
                    help="At least 6 characters.",
                )
                confirm = st.text_input("Confirm password", type="password")
                created = st.form_submit_button(
                    "Create account", type="primary", width="stretch"
                )

            if created:
                if new_password != confirm:
                    st.error("The two passwords do not match.")
                else:
                    ok, message = auth.register(name, new_email, new_password)
                    (st.success if ok else st.error)(message)
