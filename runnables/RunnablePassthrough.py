from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1
)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="write a 20 words joke on {topic}",
    input_varibles=["topic"]
)

prompt2=PromptTemplate(
    template="explain the {joke}",
    input_varibles=["joke"]
)

joke_chain=RunnableSequence(prompt1,model,parser)

chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    "explain":RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_chain,chain)

result=final_chain.invoke({"topic":"misogynists"})
print(result)

final_chain.get_graph().print_ascii()
