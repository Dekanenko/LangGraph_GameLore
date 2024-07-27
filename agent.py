from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.chains.llm import LLMChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from pydantic import BaseModel, Field
from pydantic import validator
from pathlib import Path
from typing import Dict, TypedDict
import numpy as np


class GraphState(TypedDict):
    keys: Dict[str, any]

class GraphModel():

    def __init__(self, retriever, llm, up_bound=0.8, low_bound=0.4):
        self.retriever = retriever
        self.up_bound = up_bound
        self.low_bound = low_bound
        self.llm = llm

        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("grade_docs", self.grade_docs)
        graph.add_node("generate", self.generate)
        graph.add_node("craft_search_query", self.craft_search_query)
        graph.add_node("web_search", self.web_search)
        graph.add_node("refine_docs", self.refine_docs)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_docs")
        graph.add_conditional_edges(
            "grade_docs", 
            self.decide_to_regenerate,
            {
                0: "refine_docs",
                1: "craft_search_query",
                2: "craft_search_query",
            })
        graph.add_edge("craft_search_query", "web_search")
        graph.add_edge("web_search", "refine_docs")
        graph.add_edge("refine_docs", "generate")
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

        def get_page_content(documents):
            out = "\n".join(doc.page_content for doc in documents)
            return out
        
        ethical_principle = ConstitutionalPrinciple(
            name="Ethical Principle",
            critique_request="The model should be polite and never insult the user",
            revision_request="Rewrite the model's output to be polite.",
        )

        game_image_principle = ConstitutionalPrinciple(
            name="Game Image Principle",
            critique_request="The model should never say anything bad about the Genshin Impact game. It can say bad things about characters, but it should not critique the game itself",
            revision_request="Rewrite the model's output discarding all the bad information about the Genshin Impact game.",
        )

        prompt = PromptTemplate(input_variables=["context", "question"],  template=Path("prompts/generation.prompt").read_text())

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=StrOutputParser()
        )

        constitutional_chain = ConstitutionalChain.from_llm(
            llm=self.llm,
            prompt=prompt,
            chain=chain,
            constitutional_principles=[ethical_principle, game_image_principle],
            verbose=False
        )

        generation = constitutional_chain.run(context=get_page_content(documents), question=question)

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
        retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=self.llm)

        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(input_variables=["document", "question"],  
                                template=Path("prompts/grader.prompt").read_text(),
                                partial_variables={"format_instructions": output_parser.get_format_instructions()})

        chain = prompt | self.llm

        filtered_docs = []        
        grades = []
        for doc in documents:
            try:
                response = chain.invoke({"document": doc.page_content, "question": question})
                grade = output_parser.parse(response.content)
            except:
                grade = retry_parser.parse_with_prompt(response.content, prompt.format_prompt(document=doc.page_content, question=question))

            grades.append(grade.grade)
            if grade.grade >= self.low_bound:
                filtered_docs.append(doc)
        
        grades = np.array(grades)
        return {"keys": {"documents": filtered_docs, "question": question, "grades": grades}}
    
    def decide_to_regenerate(self, state):
        grades = state["keys"]["grades"]

        # 0 - correct; 1 - incorrect; 2 - ambiguous
        strategy = 2

        if (grades > self.up_bound).any():
            strategy = 0
        elif (grades < self.low_bound).all():
            strategy = 1
        
        return strategy
    
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
    
    def refine_docs(self, state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        compressor = LLMChainExtractor.from_llm(self.llm)
        refined_docs = compressor.compress_documents(documents=documents, query=question)

        return {"keys": {"question": question, "documents": refined_docs}}