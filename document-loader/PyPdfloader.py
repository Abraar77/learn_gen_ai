from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("cv.pdf")

docs=loader.load()

print(docs[1].page_content) #every page is a seperate document
# print(docs[0])
# print(type(docs[0]))
# print(type(docs))
# print(len(docs))
# print(type(docs[0]))