## App.py
import streamlit as st
import os
import sys
print("CWD:", os.getcwd())
print("PATH:", sys.path[:5]) 

from modules import page1, page2


def main():

	with st.sidebar:
		st.session_state.type_of_query = st.sidebar.selectbox(
			"How do you want search an outfit?",
			("One vector", "Hybrid"),
			index= 0
		)

	st.set_page_config(page_title="My Mirror on Cloud", layout="wide")
	tab1, tab2, tab3 = st.tabs(["App", "More information", "ChatBot"])
	with tab1:
		page1.show()

	with tab2:
		page2.show()

if __name__ == "__main__":
    main()
