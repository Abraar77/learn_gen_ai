from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model= ChatGoogleGenerativeAI(
      model="gemini-3.5-flash-lite",
      temperature=0.1
)

parser=StrOutputParser()

prompt1= PromptTemplate(
    template="give me the best argument for {religion} being true in 50 words",
    input_varibles=["religion"]
)

prompt2=PromptTemplate(
    template="best critique atheist give to this {religion}",
    input_varibles=["religion"]
)

parllel_chain= RunnableParallel({
    "support": RunnableSequence(prompt1,model,parser),
    "critique": RunnableSequence(prompt2,model,parser)
})

result=parllel_chain.invoke({"religion":"islam"})
print(result["support"])
print(result["critique"])


parllel_chain.get_graph().print_ascii()

