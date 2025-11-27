import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page


choice = st.sidebar.selectbox("Explore or Predict", options=[
   "Explore Data",  "Predict SW developer salary"])

if choice == "Explore Data":
    show_explore_page()
else:
    show_predict_page()