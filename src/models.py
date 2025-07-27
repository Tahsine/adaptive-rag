from os import getenv
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=getenv("GOOGLE_GEMINI_API_KEY"),
)

embd = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=getenv("GOOGLE_GEMINI_API_KEY")
)

search = TavilySearch(
    api_key=getenv("TAVILY_API_KEY"),
    max_results=3
)