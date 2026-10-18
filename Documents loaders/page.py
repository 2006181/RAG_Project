from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = WebBaseLoader("https://www.groq.com/blog/introducing-groq")
docs = data.load()


ChatPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizes the website content"),
        ("human", "{data}"),
    ]
)

model = model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=1024
)

prompt = ChatPrompt.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)