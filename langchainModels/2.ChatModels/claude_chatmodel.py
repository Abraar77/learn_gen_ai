from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv


load_dotenv()

model= ChatAnthropic(model="model_name")

result=model.invoke("5 lines about kashmir")

print(result.content)

