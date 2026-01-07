import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"

# 1. Load all text files
documents = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_DIR, filename), encoding="utf-8")
        documents.extend(loader.load())

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings (Sentence Transformers)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Build FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save index
vectorstore.save_local("faiss_index")

print("Vector index built and saved!")
