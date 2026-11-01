from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system',"you are a helpul assistant in {domain}"),
    ('human', 'explain the {topic} in laymen language')
])

prompt=chat_template.invoke({
    "domain":"cricket",
    "topic":"dosra"
})

print(prompt)