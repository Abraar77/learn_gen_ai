from dotenv import load_dotenv
from langchain_classic.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

video_id= str(input("enter the video id of youtube video:"))
language=str(input("enter the language of a transcript:"))

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=[language])
    transcript=" ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("video have no caption")

# # print(transcript)
# print(type(transcript))
# # print(type(transcript))

#TEXT-SPLITTING

splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunk= splitter.create_documents([transcript])

#embedding and storing it in vector store

load_dotenv()
embedding_model=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store= Chroma.from_documents(chunk,embedding_model)

#retriever
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
# print(retriever.invoke("what is this video about?"))



llm= ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question=str(input("ask any question about the video:"))
# retrieved_docs=retriever.invoke(question)

def format_docs(docs):
    context_text="\n\n".join(doc.page_content for doc in docs)
    return context_text

paralell_chain=RunnableParallel({
    "context": retriever|RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})

parser=StrOutputParser()

#Augumentation
main_chain=paralell_chain|prompt|llm|parser

result=main_chain.invoke(question)
print(result)