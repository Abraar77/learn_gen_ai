from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path="book",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs=loader.load()

print(len(docs))
# print(docs[1].page_content)
# for document in docs:
#     print(document.metadata)