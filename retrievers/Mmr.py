#Maximal Marginal Relevance (MMR)

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

embedding_model= GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

all_docs=[
    
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store=Chroma.from_documents( # using .from_document you can add documents here without seperatively writing code for it like .add_documments
    documents=all_docs,
    embedding=embedding_model,
    collection_name="my-collection"
    
)

retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":2})

query="what is chromadb?"

result= retriever.invoke(query)

for doc in result:
    print(doc)