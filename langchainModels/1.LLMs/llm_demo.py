from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5")

result=llm.invoke("five lines about kashmir")

print(result)