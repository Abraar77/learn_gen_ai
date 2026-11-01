# from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

# import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

# os.environ['HF_HOME']='D:/huggingface_cache'

# llm=HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#             task='text-generation',
#             pipeline_kwargs=dict(
#                 temperature=0.5,
#                 max_new_tokens=200
#             )
# )

# model=ChatHuggingFace(llm=llm)

# model = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     temperature=0.1
# )

# model = ChatAnthropic(
#     model="claude-opus-5",  # Use the exact model ID in your console
#     anthropic_api_key=os.getenv("AGENTROUTER_API_KEY"),
#     base_url="https://agentrouter.org",
#     max_tokens=512,
# )

model = ChatOpenAI(
    model="gpt-6-astra",
    api_key=os.getenv("AGENTROUTER_API_KEY"),
    base_url="https://agentrouter.org/v1",
    max_tokens=512,
)


class person(BaseModel):
    name:str=Field(description="enter the name")
    age:int=Field(description="enter the age of person")
    city:str=Field(description="enter the city")

parser= PydanticOutputParser(pydantic_object=person)

template=PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)    

chain=template|model|parser

final_result=chain.invoke({"place":"kashmir"})

print(final_result)