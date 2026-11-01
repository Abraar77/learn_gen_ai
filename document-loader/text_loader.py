from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(
      model="gemini-3.5-flash-lite",
      temperature=0.1
)

prompt=PromptTemplate(
    template='answer all my quires-->{queries} on this document-->{document}, queries is a list of quries and answers use all elements of list as context just the last element as query',
    input_varibles=["queries","document"]
)

docs=TextLoader('plain.txt', encoding='utf-8')

d=docs.load()
def inputt():
    query=input("enter the query:")
    return query
parser=StrOutputParser()
chain=prompt|model|parser
inp=[]
n=0
while(n==0):
    query=input("enter the query:")
    inp.append(query)
    
    result=chain.invoke({"queries":inp,"document":d})
    inp.append(result)
    print(result)
    n=int(input("enter 0 to continue asking else any number other than:"))


print(inp)
# # print(d[0].page_content)
# print(d[0].metadata)
# print(type(d))
# print(len(d))