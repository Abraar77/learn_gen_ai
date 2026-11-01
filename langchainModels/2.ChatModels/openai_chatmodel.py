from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatmodel=ChatOpenAI(model="model name")

result=chatmodel.invoke("5 lines about kashmir")

print(result.content)