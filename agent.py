"""
Module for the stateful graph-based agent.
"""

from pathlib import Path
from typing import Dict, TypedDict, List

import numpy as np
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator

GENERATION_PROMPT_PATH = Path("prompts/generation.prompt")
GRADER_PROMPT_PATH = Path("prompts/grader.prompt")
CRAFT_SEARCH_QUERY_PROMPT_PATH = Path("prompts/craft_search_query.prompt")


# --- Pydantic Models for Output Parsing ---
class RelevanceGrade(BaseModel):
    """Pydantic model for grading the relevance of a document."""

    grade: float = Field(
        description="Grade describing document relevance; range [0, 1]", ge=0, le=1
    )

    @field_validator("grade")
    def check_grade_boundaries(cls, score: float) -> float:
        """Validate that the grade is within the [0, 1] range."""
        if not 0 <= score <= 1:
            raise ValueError("Generated grade does not fall into the [0, 1] range")
        return score


# --- Graph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary to hold the state variables.
    """

    keys: Dict[str, any]


# --- Agent Class ---
class GraphModel:
    """
    A graph-based agent that answers questions using a retrieval-augmented generation (RAG) approach.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        up_bound: float = 0.8,
        low_bound: float = 0.4,
    ):
        """
        Initializes the GraphModel.

        Args:
            retriever: The retriever to use for fetching documents.
            llm: The language model to use for generation and grading.
            up_bound: The upper bound for document relevance grades.
            low_bound: The lower bound for document relevance grades.
        """
        self.retriever = retriever
        self.llm = llm
        self.up_bound = up_bound
        self.low_bound = low_bound

        # Initialize chains and tools
        self._initialize_chains()

        self.model = self._build_graph()

    def _initialize_chains(self):
        """Initializes all the chains and tools required for the agent."""
        # Generation chain
        ethical_principle = ConstitutionalPrinciple(
            name="Ethical Principle",
            critique_request="The model should be polite and never insult the user.",
            revision_request="Rewrite the model's output to be polite.",
        )
        game_image_principle = ConstitutionalPrinciple(
            name="Game Image Principle",
            critique_request="The model should not say anything bad about the Genshin Impact game.",
            revision_request="Rewrite the model's output to be positive about the game.",
        )
        generation_prompt = PromptTemplate.from_template(
            template=GENERATION_PROMPT_PATH.read_text()
        )
        llm_chain = LLMChain(
            llm=self.llm, prompt=generation_prompt, output_parser=StrOutputParser()
        )
        self.generation_chain = ConstitutionalChain.from_llm(
            llm=self.llm,
            chain=llm_chain,
            constitutional_principles=[ethical_principle, game_image_principle],
            verbose=False,
        )

        # Document grading chain
        output_parser = PydanticOutputParser(pydantic_object=RelevanceGrade)
        grading_prompt = PromptTemplate(
            input_variables=["document", "question"],
            template=GRADER_PROMPT_PATH.read_text(),
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        self.grading_chain = grading_prompt | self.llm
        self.grading_parser = output_parser

        # Query crafting chain
        craft_query_prompt = PromptTemplate.from_template(
            template=CRAFT_SEARCH_QUERY_PROMPT_PATH.read_text()
        )
        self.craft_query_chain = craft_query_prompt | self.llm | StrOutputParser()

        # Web search tool
        self.web_search_tool = TavilySearchResults()

        # Document refinement compressor
        self.refine_compressor = LLMChainExtractor.from_llm(self.llm)

    def _build_graph(self) -> StateGraph:
        """Builds and compiles the langgraph state machine."""
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
                0: "refine_docs",  # All good, refine and generate
                1: "craft_search_query",  # All bad, new search
                2: "craft_search_query",  # Mixed, new search
            },
        )
        graph.add_edge("craft_search_query", "web_search")
        graph.add_edge("web_search", "refine_docs")
        graph.add_edge("refine_docs", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def retrieve(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Retrieves documents based on the question."""
        question = state["keys"]["question"]
        documents = self.retriever.invoke(question)
        return {"keys": {"documents": documents, "question": question}}

    def generate(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Generates an answer using the retrieved documents."""
        question = state["keys"]["question"]
        documents: List[Document] = state["keys"]["documents"]
        context = "\n".join(doc.page_content for doc in documents)

        generation = self.generation_chain.run(context=context, question=question)
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation,
            }
        }

    def grade_docs(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Grades the relevance of retrieved documents."""
        question = state["keys"]["question"]
        documents = state["keys"]["documents"]

        filtered_docs = []
        grades = []
        for doc in documents:
            response = self.grading_chain.invoke(
                {"document": doc.page_content, "question": question}
            )
            grade = self.grading_parser.parse(response.content)
            grades.append(grade.grade)
            if grade.grade >= self.low_bound:
                filtered_docs.append(doc)

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "grades": np.array(grades),
            }
        }

    def decide_to_regenerate(self, state: GraphState) -> int:
        """Determines the next step based on document grades."""
        grades = state["keys"]["grades"]
        if (grades > self.up_bound).any():
            return 0  # At least one good document
        if (grades < self.low_bound).all():
            return 1  # All documents are bad
        return 2  # Mixed or ambiguous grades

    def craft_search_query(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Improves the user's question for better web search results."""
        question = state["keys"]["question"]
        improved_question = self.craft_query_chain.invoke({"question": question})
        return {"keys": {**state["keys"], "question": improved_question}}

    def web_search(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Performs a web search and adds the results to the documents."""
        question = state["keys"]["question"]
        documents = state["keys"]["documents"]
        web_results = self.web_search_tool.invoke({"query": question})
        web_content = "\n".join(doc["content"] for doc in web_results)
        documents.append(Document(page_content=web_content))
        return {"keys": {"documents": documents, "question": question}}

    def refine_docs(self, state: GraphState) -> Dict[str, Dict[str, any]]:
        """Compresses and refines documents to extract relevant information."""
        question = state["keys"]["question"]
        documents = state["keys"]["documents"]
        refined_docs = self.refine_compressor.compress_documents(
            documents=documents, query=question
        )
        return {"keys": {"question": question, "documents": refined_docs}}
