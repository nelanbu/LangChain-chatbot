# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama

import streamlit as st
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please reponse to the user queries."),
        ("user", "Question:{question}")
    ]
)

st.title("My chatbot")
input_text = st.text_input("Ask me!")

llm = ollama.Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))