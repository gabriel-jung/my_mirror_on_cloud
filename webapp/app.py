## App.py
import streamlit as st
import os
import sys

from modules import page1
from loguru import logger

logger.info("CWD:", os.getcwd())
logger.info("PATH:", sys.path[:5])


def main():
    with st.sidebar:
        st.session_state.type_of_query = st.sidebar.selectbox(
            "How do you want search an outfit?", ("One vector", "Hybrid"), index=0
        )
    st.set_page_config(page_title="My Mirror on Cloud", layout="wide")
    page1.show()
    # tab1, tab2 = st.tabs(["App", "More information"])
    # with tab1:
    #     page1.show()

    # with tab2:
    #     page2.show()

    # with tab3:
    #     chatbot.show_chatbot()


if __name__ == "__main__":
    main()
