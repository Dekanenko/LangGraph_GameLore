"""
Main application file for the Chainlit chat interface.
"""

import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from agent import GraphModel
from data_prep import get_vector_db

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
AUTHOR_NAME = "Archivist Ramiel"
WELCOME_MESSAGE = (
    "Greetings, traveler. I am Archivist Ramiel, the keeper of all knowledge "
    "within the world of Teyvat. My vast compendium spans the intricate lore, "
    "untold stories, and hidden secrets of Genshin Impact. Unlike the fragile "
    "records of Irminsul, my knowledge remains eternal and unblemished. "
    "How may I assist you on your quest for wisdom today?"
)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
retriever = get_vector_db(embedding_function).as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.01, "k": 5},
)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
graph_agent = GraphModel(retriever=retriever, llm=llm)


@cl.on_chat_start
async def main():
    """
    Initializes the chat session by setting up the model, retriever, and agent.
    Sends a welcome message to the user.
    """

    await cl.Message(
        content=WELCOME_MESSAGE,
        author=AUTHOR_NAME,
    ).send()


@cl.on_message
async def message(message: cl.Message):
    """
    Handles incoming user messages.
    Retrieves the agent from the user session and invokes the model.
    Sends the model's response back to the user.
    """
    user_input = message.content

    response = graph_agent.model.invoke({"keys": {"question": user_input}})["keys"][
        "generation"
    ]
    await cl.Message(
        content=response,
        author=AUTHOR_NAME,
    ).send()
