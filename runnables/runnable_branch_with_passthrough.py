from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()
model= ChatGoogleGenerativeAI(
      model="gemini-3.5-flash-lite",
      temperature=0.1
)

class sentiment(BaseModel):
    sentiment:Literal['positive',"negative"]=Field(description="give the sentiment of feedback either positive or negative")

parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=sentiment)

prompt1= PromptTemplate(
    template="determine the sentiment of feedback-->{feedback} in {format_instruction}",
    input_varibles=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

prompt2= PromptTemplate(
    template="generate the professional response for this positive feedback--->{feedback}",
    input_varibles=["feedback"],
    
)    
prompt2= PromptTemplate(
    template="generate the professional response for this positive feedback--->{feedback}",
    input_varibles=["feedback"],
    
)    
prompt3= PromptTemplate(
    template="generate the professional response for this negative feedback--->{feedback}",
    input_varibles=["feedback"],
    
)    

sentiment_chain=RunnableSequence(prompt1,model,parser2)

branch_chain=RunnableBranch(
    (lambda x:x["sentiment"].sentiment=="postive", RunnableSequence(prompt2,model,parser1)),
    (lambda x:x["sentiment"].sentiment=="negative", RunnableSequence(prompt3,model,parser1)),
    RunnableLambda(lambda x:x.sentiment=="sentiment can't be detected")
)

final_chain=RunnableSequence(RunnablePassthrough.assign(sentiment=sentiment_chain), RunnablePassthrough.assign(response=branch_chain))

result=final_chain.invoke({"feedback":"the screen is too big"})
print("Feedback:", result["feedback"])
print("Sentiment:", result["sentiment"].sentiment)
print("Response:", result["response"])

