from langchain_core.prompts import ChatPromptTemplate

from .utils import related_docs

# Prompts 
system_router_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to {related_docs}.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

system_retrieval_grader_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

system_hallucination_grader_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

system_answer_grader_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

system_question_rewriter_prompt = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""


# Chat Prompt Templates
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router_prompt.format(related_docs=related_docs)),
        ("human", "{question}"),
    ]
)

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_retrieval_grader_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination_grader_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer_grader_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

question_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_question_rewriter_prompt),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

human_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """
        )
    ]
)
