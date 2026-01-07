# query.py
# RAG chatbot for 3 machine translation papers using FAISS and MPT-7B-8k-Instruct

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# =====================
# 1️⃣ Load embeddings
# =====================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================
# 2️⃣ Load FAISS index
# =====================
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # safe since we created this index
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =====================
# 3️⃣ Load MPT-7B-8k-Instruct LLM (GPU-optimized)
# =====================
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

hf_pipeline = pipeline(
    "text-generation",
    model=model_name,
    max_new_tokens=300,
    do_sample=False,
    temperature=1.0,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# =====================
# 4️⃣ Define improved prompt with citations
# =====================
template = """Use the following context to answer the question.
Cite the source of each fact in parentheses with the filename.
Do not copy citations like [15] or [16] from the text.
If you don't know the answer, say you don't know. Answer in clear, complete sentences.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# =====================
# 5️⃣ Create RetrievalQA chain
# =====================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # stuff all chunks into the prompt
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# =====================
# 6️⃣ Ask a question
# =====================
question = "Why is self-attention important for machine translation?"
result = qa_chain({"query": question})


raw_answer = result["result"]

# Split at the last occurrence of "Answer:" and take what comes after
answer = raw_answer.split("Answer:")[-1].strip()

# =====================
# 7️⃣ Print answer and sources
# =====================
print("Answer:\n", answer)
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown"))


'''
Improvements:
- better chunking
- improve prompt
- add full comments
- slow generating answers?
- preporcess docs more
- add in csv/more docs

Current output:
Answer:
 Self-attention is important for machine translation because it has been used successfully in a variety of tasks, including machine translation, and it can help the Transformer model to achieve a new state of the art on both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks (source: context).

Sources:
- data\attention_is_all_you_need.txt
- data\attention_is_all_you_need.txt
- data\human_parity.txt

'''

