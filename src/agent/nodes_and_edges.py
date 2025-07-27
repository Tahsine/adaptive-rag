from langchain.schema import Document

from .state import GraphState
from src.retriever import retriever
from src.utils import format_docs

from .agents_utils import retrieval_grader, question_rewriter, web_search_tool, question_router, hallucination_grader, answer_grader,  rag

# Node
def retrieve(state: GraphState):
    """Retrieve documents.
        Args:
            state (GraphState): The current graph state.
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("Retrieving documents...")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (GraphState): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    
    """
    print("Generating answer...")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    docs_txt = format_docs(documents)
    generation = rag.invoke({"context": docs_txt, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: GraphState):
    """
    Grade documents relevance

    Args:
        state (GraphState): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains graded documents
    """
    print("Grading documents...")
    question = state["question"]
    documents = state["documents"]

    # Score documents
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.
    
    Args:
        state (GraphState): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("Transforming query...")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state: GraphState):
    """
    Web search based on the re-phrased question.

    Args:
        state (GraphState): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("Web searching...")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs['results']])
    web_results = [Document(page_content=web_results)]

    return {"documents": web_results, "question": question}


# Edges

def route_question(state: GraphState):
    """
    Route question to web search or RAG.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Next node to call
    """
    print("Routing question...")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state: GraphState):
    """
    Decide whether to generate an answer, or re-generate a question.
    
    Args:
        state (GraphState): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("Access graded documents...")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print(
            "---NO RELEVANT DOCUMENTS FOUND, GENERATING NEW QUESTION---"
        )
        return "transform_query"
    else:
        print("---RELEVANT DOCUMENTS FOUND, GENERATING ANSWER---")
        return "generate"
    
def grade_generation_vs_documents_and_question(state: GraphState):
    """
    Determine wheter the generation is grounded in the documents and answers question.
    
    Args:
        state (GraphState): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("Checking hallucinations...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
