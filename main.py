from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriver = vectorstore.as_retriever(
 search_type = "mmr",
 search_kwargs = {
  "k" : 4,
  "fetch_k" : 10,
  "lambda_mult" : 0.5 
 }
)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

#prompt template 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful Ai assistant.messages.messages.

         Use only the provided context to answer the question.messages=
         If the answer is not present in the context,
         say: "I could not find the answer in the document."
         """

        ),
        (
            "human",
            """Context:
            {context}
            
            Question:
            {question}"""
        )
    ]
)

print("rag system created")

print("press 0 to exit")

while True:
    query = input("You")
    if query == "0":
        break
    
    docs = retriver.invoke(query)

    context = "".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")

