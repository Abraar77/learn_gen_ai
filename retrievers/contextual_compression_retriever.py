from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
load_dotenv()

llm= ChatGoogleGenerativeAI(
   model="gemini-3.5-flash-lite",
#    temperature=0.1
)

embedding_model=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

vector_store= Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="ccr"
)

compressor= LLMChainExtractor.from_llm(llm)

retriever=vector_store.as_retriever(
    search_kwargs={"k":2}
)

compression_retriever= ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

query="what is photosynthesis?"

result= compression_retriever.invoke(query)

for i, docs in enumerate(result):
    print(f"Result {i+1} \n")
    print(docs.page_content) 