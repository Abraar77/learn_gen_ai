from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1
)


parser=StrOutputParser()

prompt1= PromptTemplate(
    template="write a joke on this {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="explain the {joke}",
    input_variables=["joke"]
)

chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({"topic":"feminism"})
print(result)