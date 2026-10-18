from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("Documents loaders/GRU.pdf")

docs = data.load()

print(docs)