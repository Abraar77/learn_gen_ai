from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
os.environ['HF_HOME'] = 'D:/huggingface_cache'
# load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200
    )
)
parser = StrOutputParser()
template = PromptTemplate(
    template="Write five facts about {topic} ",
    input_variables=["topic"]
    
)



model = ChatHuggingFace(llm=llm)

chain=template|model|parser
final_result=chain.invoke({"topic":"black hole"})

# prompt=template.invoke({"topic":"black hole"})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

print(final_result)