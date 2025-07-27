from langchain_core.output_parsers import StrOutputParser

from src.models import llm, search
from src.classes import RouteQuery, GradeDocuments, GradeHallucinations, GradeAnswer
from src.prompts import router_prompt, retrieval_grader_prompt, hallucination_prompt, answer_grader_prompt, question_rewriter_prompt, human_prompt

# Strutured LLM
structured_llm_router = llm.with_structured_output(RouteQuery)
structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

# Chain
question_router = router_prompt | structured_llm_router
retrieval_grader = retrieval_grader_prompt | structured_llm_doc_grader
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
answer_grader = answer_grader_prompt | structured_llm_answer_grader
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

rag = human_prompt | llm | StrOutputParser()

# Web Search Tool
web_search_tool = search


