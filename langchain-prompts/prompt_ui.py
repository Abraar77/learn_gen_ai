from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
    task="text-generation"
    
)
template=load_prompt("template.json")
st.header("Research tool")

model = ChatHuggingFace(llm=llm)
paper_input= st.selectbox("select the research paper",["Attention is all you need","Bert","Gpt2","sequence to sequence learning"])
explain_type=st.selectbox("select the explain type",["code oreiented","mathamatically","begginer friendly"])
length_input=st.selectbox("select the length",["20-50 words", "50-100 words","100-200 words"])

if st.button("Summerise"):
    chain=template|model
    result=chain.invoke({
         "length_input":length_input,
         "paper_input":paper_input,
         "style_input":explain_type
    })
    st.write(result.content)
