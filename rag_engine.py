import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

def load_and_index_pdf(pdf_path, hf_token):
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
        model_name="all-MiniLM-L6-v2"
    )

    print("Vector database ban raha hai...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Done! ✅")
    return vectorstore, embeddings

def get_qa_chain(vectorstore, hf_token):
    client = InferenceClient(
        model="Qwen/Qwen2.5-72B-Instruct",
        token=hf_token
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def answer_question(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are a Pakistan laws expert. Respond in SAME language as question.
- Urdu question → Urdu answer
- English question → English answer
- Roman Urdu → Roman Urdu answer

Context:
{context}

Question: {question}

Answer:"""
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return response.choices[0].message.content

    return answer_question