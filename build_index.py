from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
from dotenv import load_dotenv
import os
import chromadb

load_dotenv(override=True)  # Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
def build_chroma_index(docs, document_hash):
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    Settings.llm   = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", embed_batch_size=100
    )
    """
    Build a LlamaIndex that uses Chroma as the underlying vector store.
    """
    # 1) Create a ChromaVectorStore
    #    By default, ChromaVectorStore will either run in-memory 
    #    or require a 'persist_directory' for disk-based storage.
    # Use a local Chroma settings object
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")

    chroma_collection = chroma_client.get_or_create_collection(name=document_hash)

        # Check if the collection has existing records (i.e., document is already indexed)
    if chroma_collection.count() > 0:
        print(f"Document {document_hash} found in Chroma, loading existing index...")
        
        # Load existing index from storage
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = load_index_from_storage(storage_context)

    else:
        print(f"Document {document_hash} not found in Chroma, building new index...")
        
        # Create a new vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build an index from the provided documents
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    return index

if __name__ == "__main__":
    from parse_document import parse_pdf

    file_path = "uploads/RAG_target_file.pdf"
    docs, document_hash = parse_pdf(file_path, chunk_size=1000, overlap=100)

    # Build the index with Chroma
    index = build_chroma_index(docs, document_hash)