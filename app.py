from langchain_openai import ChatOpenAI
from data_prep import get_vector_db
from agent import GraphModel
import chainlit as cl
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from dotenv import load_dotenv
load_dotenv()


embedding_function = embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = get_vector_db(embedding_function).as_retriever(search_type="similarity_score_threshold", 
                                              search_kwargs={"score_threshold": 0.01, "k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
graph_agent = GraphModel(retriever=retriever, llm=llm)

@cl.on_chat_start
async def main():
    await cl.Message(
        content="Greetings, traveler. I am Archivist Ramiel, the keeper of all knowledge within the world of Teyvat. My vast compendium spans the intricate lore, untold stories, and hidden secrets of Genshin Impact. Unlike the fragile records of Irminsul, my knowledge remains eternal and unblemished. How may I assist you on your quest for wisdom today?",
        author="Archivist Ramiel"
    ).send()


@cl.on_message
async def message(message: str):
    user_input = message.content

    response = graph_agent.model.invoke({"keys": {"question": user_input}})["keys"]["generation"]
    await cl.Message(
        content=response,
        author="Archivist Ramiel"
    ).send()
