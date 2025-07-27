from langchain_chroma import Chroma

from .models import embd
from .utils import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH

try:
    vectorstore = Chroma(
        embedding_function=embd,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print(f"Chroma DB charged from {CHROMA_DB_PATH}")

except Exception as e:
    vector_store = None
    raise f"Error loading Chroma DB: {e}"

if vectorstore:
    retriever = vectorstore.as_retriever()
else:
    retriever = None


if __name__ == "__main__":
    if retriever:
        print("Retriever is ready to use.")
    else:
        print("Retriever could not be initialized.")