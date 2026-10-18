from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

data = TextLoader("Documents loaders/text.txt", encoding="utf-8")
documents = data.load()

splitter = TokenTextSplitter(separator="\n\n",chunk_size=1000, chunk_overlap=1)
chunks = splitter.split_documents(documents)
print(chunks[0].page_content)