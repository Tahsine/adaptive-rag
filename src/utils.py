CHROMA_COLLECTION_NAME = "adaptive-rag-chroma"
CHROMA_DB_PATH = "db/adaptive-rag-chroma-db"

related_docs = """agents, prompt engineering, and adversarial attacks"""


# Post-processing
def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)