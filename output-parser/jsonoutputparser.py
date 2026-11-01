from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
os.environ['HF_HOME']='D:/huggingface_cache'

llm= HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=200
        )
)

parser = JsonOutputParser()

template= PromptTemplate(
    template="Give exactly five facts about {topic}.\n"
        "Return ONLY a valid JSON object in this format:\n"
        '{{"facts": ["fact 1", "fact 2", "fact 3", '
        '"fact 4", "fact 5"]}}\n'
        "Do not include markdown, explanations, or text outside JSON.",
    input_variables=["topic"],
    # partial_variables={'format_instruction':parser.get_format_instructions()}
)

model= ChatHuggingFace(llm=llm)

chain=template|model|parser

final_result=chain.invoke({"topic":"black hole"})

print(final_result)
