from rag_engine import load_and_index_pdf, get_qa_chain

API_KEY = "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"

vectorstore, embeddings = load_and_index_pdf("data/labor_law.pdf")
answer = get_qa_chain(vectorstore, API_KEY)

result = answer("Pakistan mein minimum wage kya hai?")
print(result)