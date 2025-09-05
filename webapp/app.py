## App.py
import streamlit as st
import os, sys
print("CWD:", os.getcwd())
print("PATH:", sys.path[:5]) 

from modules import page1, page2



def main():

	
	st.set_page_config(page_title="My Mirror on Cloud", layout="wide")
	tab1, tab2 = st.tabs(["App", "More information"])
	with tab1:
		page1.show()

	with tab2:
		page2.show()


if __name__ == "__main__":
    main()
