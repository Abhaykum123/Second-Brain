from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
)

from backend.rag.vectorstore import similarity_search

from backend.rag.prompts import (
    QUERY_REWRITE_PROMPT,
    GRADING_PROMPT,
    ANSWER_PROMPT,
)

# GROQ LLM
llm = ChatGroq(
    model=GROQ_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=2048,
)

# RAG STATE
class RAGState(TypedDict):
    question: str
    rewritten_query: str
    documents: List[Document]
    relevant_documents: List[Document]
    answer: str


# RESPONSE TEXT EXTRACTION
def get_response_text(response) -> str:
    """
    Safely extract text from Groq/LangChain response.
    """

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

        return "".join(parts)

    return str(content)


# QUERY REWRITING
def rewrite_query(state: RAGState):

    try:

        prompt = QUERY_REWRITE_PROMPT.invoke({
            "question": state["question"]
        })

        response = llm.invoke(prompt)

        query = get_response_text(response).strip()

        # Fallback to original question
        if not query:
            query = state["question"]

        return {
            "rewritten_query": query
        }

    except Exception as e:

        print(
            "QUERY REWRITE ERROR:",
            repr(e)
        )

        # If Groq fails, use original question
        return {
            "rewritten_query": state["question"]
        }


# DOCUMENT RETRIEVAL
def retrieve_documents(state: RAGState):

    documents = similarity_search(
        state["rewritten_query"],
        k=TOP_K
    )

    return {
        "documents": documents
    }


# DOCUMENT GRADING
def grade_documents(state: RAGState):

    relevant_documents = []

    for document in state["documents"]:

        try:

            prompt = GRADING_PROMPT.invoke({
                "question": state["question"],
                "document": document.page_content,
            })

            response = llm.invoke(prompt)

            result = get_response_text(
                response
            ).strip().upper()

            # Accept documents when Groq returns YES
            if result.startswith("YES"):

                relevant_documents.append(
                    document
                )

        except Exception as e:

            print(
                "DOCUMENT GRADING ERROR:",
                repr(e)
            )

            # Continue grading remaining documents
            continue

    return {
        "relevant_documents": relevant_documents
    }


# ANSWER GENERATION
def generate_answer(state: RAGState):

    documents = state["relevant_documents"]

    # No relevant documents
    if not documents:

        return {
            "answer": (
                "I don't have enough information "
                "in your knowledge base to answer "
                "this question."
            )
        }

    # Build context
    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "file_name",
            document.metadata.get(
                "source",
                "unknown"
            )
        )

        context_parts.append(
            f"Source: {source}\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # Generate answer with Groq
    try:

        prompt = ANSWER_PROMPT.invoke({
            "question": state["question"],
            "context": context,
        })

        response = llm.invoke(prompt)

        answer = get_response_text(
            response
        ).strip()

        if not answer:

            return {
                "answer": (
                    "I could not generate an answer "
                    "from the retrieved information."
                )
            }

        return {
            "answer": answer
        }

    except Exception as e:

        print(
            "ANSWER GENERATION ERROR:",
            repr(e)
        )

        raise


# BUILD LANGGRAPH
def build_graph():

    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node(
        "rewrite_query",
        rewrite_query
    )

    workflow.add_node(
        "retrieve",
        retrieve_documents
    )

    workflow.add_node(
        "grade_documents",
        grade_documents
    )

    workflow.add_node(
        "generate_answer",
        generate_answer
    )

    # Add edges

    workflow.add_edge(
        START,
        "rewrite_query"
    )

    workflow.add_edge(
        "rewrite_query",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "grade_documents"
    )

    workflow.add_edge(
        "grade_documents",
        "generate_answer"
    )

    workflow.add_edge(
        "generate_answer",
        END
    )
    # Compile graph
    return workflow.compile()


# CREATE RAG GRAPH
rag_graph = build_graph()

# ASK QUESTION
def ask_question(question: str):

    result = rag_graph.invoke({

        "question": question,

        "rewritten_query": "",

        "documents": [],

        "relevant_documents": [],

        "answer": "",
    })

    return result["answer"]