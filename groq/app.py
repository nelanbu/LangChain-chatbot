## bot with groq api 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader 
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://www.britannica.com/place/Ottoman-Empire")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Learn Ottoman Empire History")
st.subheader("groq demo")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Gemma-7b-It")

prompt = ChatPromptTemplate.from_template("Answer the questions only based on the provided context.\
                                           Provide the most accurate response.\
                                           {context}\
                                           Questions: {input} ")

document_chain = create_stuff_documents_chain(llm, prompt)
ret = st.session_state.vectors.as_retriever()
ret_chan = create_retrieval_chain(ret, document_chain)

prompt = st.text_input("Ask me about Ottoman Empire!")

if prompt:
    result = ret_chan.invoke({"input": prompt})
    st.write(result["answer"])