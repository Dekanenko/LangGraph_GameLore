from langchain_openai import ChatOpenAI
from data_prep import get_vector_db
from agent import GraphModel
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()

retriever = get_vector_db().as_retriever(search_type="similarity_score_threshold", 
                                              search_kwargs={"score_threshold": 0.05, "k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
graph_agent = GraphModel(retriever=retriever, llm=llm)

@cl.on_chat_start
async def main():
    await cl.Message(
        content="Welcome to my Library, I am Adam titled 'The Greatest Librarian' here to assist you with your questions",
        author="Adam. The Greater Librarian"
    ).send()


@cl.on_message
async def main(message: str):
    user_input = message.content

    response = graph_agent.model.invoke({"keys": {"question": user_input}})["keys"]["generation"]
    await cl.Message(
        content=response,
        author="Adam. The Greater Librarian"
    ).send()
