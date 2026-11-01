from langchain_openai import OpenAiEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAiEmbeddings(model="text-embedding-3-large", dimensions=32)

document=[
    "delhi is the capital of india",
    "srinagar is the capital of jandk"
]

result=embedding.embed_documents(document)

print(str(result))