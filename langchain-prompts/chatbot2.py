from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.messages import SystemMessage,HumanMessage,AIMessage 



load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
    task="text-generation"
    
)

model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="you are a helpful assistant"),
    
]

def chat():
    inp=input("chat or exit:")
    if inp=="chat":

        pr=str(input("You:"))
        chat_history.append(HumanMessage(content=pr))
        output()
    else:
        print(chat_history)
        exit()    

def output():
    result = model.invoke(f"Ai:{chat_history}")
    chat_history.append(AIMessage(content=result.content))
    print("Ai",result.content)
    chat()


chat()
print(chat_history)