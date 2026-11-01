from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


documents = [
    
    "Kolkata is the capital of West Bengal",
    "Delhi is the capital of India",
    "Paris is the capital of France"
]
query=input("enter your query:")
docs= embedding.embed_documents(documents)
query_embed=embedding.embed_query(query)

scores=cosine_similarity([query_embed],docs)[0]

print(scores)

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]


print(f"Question:{query}. Answer: {documents[index]}")
