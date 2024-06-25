import requests
import streamlit as st

def get_llm_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json={'input' : {'topic' : input_text}})

    return response.json()['output'] #['content']


st.title("Langchain simple app with Llama2")

input_text = st.text_input("Write an essay about ...")

st.write(get_llm_response(input_text))