from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate([
    ('system', 'you are an helpul coustemer assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human',"{query}")
])
chat_history=[]
with open ('chat_history.txt') as f:
    chat_history.extend((f.readlines()))

query=input("ask:")
prompt=chat_template.invoke({
    "chat_history":chat_history,
    "query":query
})

result=model.invoke(prompt)

print(result.content)

