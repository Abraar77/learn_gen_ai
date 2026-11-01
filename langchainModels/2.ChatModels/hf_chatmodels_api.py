#USING HUGGING FACE API DIRECTLY 

# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
# load_dotenv()
# client = InferenceClient(
#     api_key=os.environ["HF_TOKEN"],
# )

# completion = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
#     messages=[
#         {
#             "role": "user",
#             "content": "what is the integration of x square?"
            
#         }
#     ],
# )

# print(completion.choices[0].message.content)


#USING LANGCHAIN

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
    task="text-generation"
    
)

model = ChatHuggingFace(llm=llm)


pr=str(input("enter the prompt here:"))
result = model.invoke(pr)

print(result.content)

