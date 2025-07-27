from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma

from .models import embd
from .utils import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH


def ingest_web_data():
    print("Ingesting web data...")

    urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Vectorstore
    vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embd,
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_DB_PATH
    )
    
if __name__ == "__main__":
    ingest_web_data()
    print("Web data ingestion completed.")
