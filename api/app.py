from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_community.llms import ollama
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "langchain server",
    version = "1.0",
    description = "simple api server"
)


# model 1
llm = ollama.Ollama(model="llama2")

# model 2
# llm = ollama.Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words.")

add_routes(
    app,
    prompt | llm,
    path = "/essay"
)

# # another route for second model can be implemented 
# add_routes(
#     app,
#     prompt | llm,
#     path = "/essay"
# )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)