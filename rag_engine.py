import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

def load_and_index_pdf(pdf_path):
    print("PDF load ho raha hai...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Chunks ban rahe hain...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print("Embeddings ban rahi hain...")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-MiniLM-L3-v2"
    )

    print("Vector database ban raha hai...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Done! Index save ho gaya ✅")
    return vectorstore, embeddings

def get_qa_chain(vectorstore, api_key):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=api_key,
        temperature=0.3,
        max_new_tokens=512
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def answer_question(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are a Pakistan laws expert. Analyze the user's question language/style and respond in the SAME language and tone:

- If question is in Urdu → answer in Urdu
- If question is in English → answer in English
- If question is in Roman Urdu → answer in Roman Urdu
- Always match user's tone

Context:
{context}

Question: {question}

Answer (same language as question):"""
        response = llm.invoke(prompt)
        return response

    return answer_question