from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from pydantic import validator
from pathlib import Path
from typing import Dict, TypedDict

class GraphState(TypedDict):
    keys: Dict[str, any]


class GraphModel():

    def __init__(self, retriever, llm, threshold=0.5):
        self.retriever = retriever
        self.threshold = threshold
        self.llm = llm

        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("grade_docs", self.grade_docs)
        graph.add_node("generate", self.generate)
        graph.add_node("craft_search_query", self.craft_search_query)
        graph.add_node("web_search", self.web_search)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_docs")
        graph.add_conditional_edges(
            "grade_docs", 
            self.decide_to_regenerate,
            {
                "generate": "generate",
                "craft_search_query": "craft_search_query"
            })
        graph.add_edge("craft_search_query", "web_search")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)

        self.model = graph.compile()

    def retrieve(self, state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.retriever.invoke(question)

        return {"keys": {"documents": documents, "question": question}}
    
    def generate(self, state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(input_variables=["context", "question"],  template=Path("prompts/generation.prompt").read_text())
        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke({"context": documents, "question": question})

        return {"keys": {"documents": documents, "question": question, "generation": generation}}
    
    def grade_docs(self, state):
        class GraderOut(BaseModel):
            grade: float = Field(desctiption="Grade that describes the document relevance; range[0, 1]",
                                 ge = 0, le = 1)

            @validator("grade", allow_reuse=True)
            def check_grade_boundaries(cls, score):
                if score > 1 or score < 0:
                    raise ValueError("Generated grade does not fall into [0, 1] range")
                return score

        output_parser = PydanticOutputParser(pydantic_object=GraderOut)

        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(input_variables=["document", "question"],  
                                template=Path("prompts/grader.prompt").read_text(),
                                partial_variables={"format_instructions": output_parser.get_format_instructions()})

        chain = prompt | self.llm | output_parser

        relevant_docs = []
        regenerate = False

        if len(documents) == 0:
            regenerate = True
        
        for doc in documents:
            grade = chain.invoke({"document": doc, "question": question, "format_instruction": output_parser.get_format_instructions()})
            if grade.grade < self.threshold:
                regenerate = True
            else:
                relevant_docs.append(doc)        

        return {"keys": {"documents": relevant_docs, "question": question, "regenerate": regenerate}}
    
    def decide_to_regenerate(self, state):
        regenerate = state["keys"]["regenerate"]
        if regenerate:
            return "craft_search_query"
        
        return "generate"
    
    def craft_search_query(self, state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(input_variables=["question"], template=Path("prompts/craft_search_query.prompt").read_text())
        chain = prompt | self.llm | StrOutputParser()
        improved_question = chain.invoke({"question": question})
        
        return {"keys": {"documents": documents, "question": improved_question}}
    
    def web_search(self, state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        search_tool = TavilySearchResults()
        web_docs = search_tool.invoke({"query": question})
        web_docs = "\n".join(doc["content"] for doc in web_docs)
        web_docs = Document(page_content=web_docs)

        documents.append(web_docs)

        return {"keys": {"documents": documents, "question": question}}
        


