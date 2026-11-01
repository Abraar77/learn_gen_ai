from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import  RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1
)

parser1=StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="generate the sentiment of this feedback-->{feedback} in {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
    
)   

sentiment_Chain=prompt1|model|parser2

prompt2=PromptTemplate(
    template="write an professional response for this positive feedback \n feedback -->{feedback}",
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="write an professional response for this negative feedback \n feedback -->{feedback}",
    input_variables=["feedback"]
)

condition_chain =RunnableBranch(
    (lambda x:x.sentiment=="positive", prompt2|model|parser1),
    (lambda x:x.sentiment=="negative", prompt3|model|parser1),
    RunnableLambda(lambda x:x=="can't detect the emotions")
)

final_Chain=sentiment_Chain|condition_chain

result=final_Chain.invoke({"feedback":"this phone is a class "})

print(result)


final_Chain.get_graph().print_ascii()

 