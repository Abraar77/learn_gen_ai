from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template1=PromptTemplate(
    template="write a poem on {place}",
    input_variables=["place"]
)

template2=PromptTemplate(
    template="write a five point summary on {poem}",
    input_variables=["poem"]
)

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1
)

parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser

result=chain.invoke({"place":"kashmir"})

print(result)


chain.get_graph().print_ascii()