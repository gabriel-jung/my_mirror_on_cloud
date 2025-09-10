## App.py
import streamlit as st
import os, sys
print("CWD:", os.getcwd())
print("PATH:", sys.path[:5]) 

from modules import page1, page2, chatbot



def main():

	
	st.set_page_config(page_title="My Mirror on Cloud", layout="wide")
	tab1, tab2, tab3 = st.tabs(["App", "More information", "ChatBot"])
	with tab1:
		page1.show()

	with tab2:
		page2.show()

	with tab3:
		chatbot.show_chatbot()


if __name__ == "__main__":
    main()
