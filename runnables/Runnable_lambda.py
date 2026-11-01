from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1
)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="write a 29 word joke on {topic}",
    input_varibles=["topic"]
)


def word_count(text):
    return len(text.split())

joke_chain=RunnableSequence(prompt1,model,parser)

parallel_chain= RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
    # "word_count":lambda x:len(x.split()) we can use this instead of runnablelambda
    
}   
)

final_chain=RunnableSequence(joke_chain,parallel_chain)

result=final_chain.invoke({"topic":"elon musk"})

print(result)