from langchain_community.retrievers import WikipediaRetriever


import wikipedia

# Set a descriptive User-Agent before using WikipediaRetriever
wikipedia.set_user_agent(
    "MyLangChainLearningProject/1.0 (contact: yourgmail@gmail.com)"
)
retriever=WikipediaRetriever(top_k_results=2, lang="en")

query="summary on first cm of kashmir"

result= retriever.invoke(query)

for doc in result:
    print(doc.page_content)