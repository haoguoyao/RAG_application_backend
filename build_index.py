from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
from dotenv import load_dotenv
import os
import chromadb
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
load_dotenv(override=True)  # Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Settings.llm   = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large", embed_batch_size=100
)
def check_chroma_index(document_hash):
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    chroma_collection = chroma_client.get_or_create_collection(name=document_hash)
    return chroma_collection.count() > 0
def build_chroma_index(docs, document_hash):
    """
    Build a LlamaIndex that uses Chroma as the underlying vector store.
    """

    # Initialize persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    chroma_collection = chroma_client.get_or_create_collection(name=document_hash)

    # Path where LlamaIndex metadata will be saved
    persist_dir = f"./chroma_storage/{document_hash}"

    # Check if the collection has existing records
    if chroma_collection.count() > 0:
        print(f"Document {document_hash} found in Chroma, loading existing index...")
        
        # Load the index from the stored metadata
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    else:
        print(f"Document {document_hash} not found in Chroma, building new index...")
        
        # Create a new vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build and persist the index
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        index.storage_context.persist(persist_dir=persist_dir)  # ✅ Persist the metadata



    return index

def get_chroma_index(document_hash):

    # Initialize Chroma Persistent Client
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    chroma_collection = chroma_client.get_or_create_collection(name=document_hash)

    # Path where LlamaIndex metadata is stored
    persist_dir = f"./chroma_storage/{document_hash}"

    # Check if the collection has existing records
    if chroma_collection.count() > 0:
        print(f"Document {document_hash} found in Chroma, loading existing index...")

        # Load stored index metadata from disk
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)

        try:
            index = load_index_from_storage(storage_context)  # ✅ Load metadata
            print(f"Index loaded successfully for document {document_hash}")
            return index
        except Exception as e:
            print(f"Error loading index: {e}")
            # print(f"persist_dir: {persist_dir}")
            return None
    else:
        print(f"Document {document_hash} not found in Chroma.")
        return None

def get_query_engine(document_hash):
    index = get_chroma_index(document_hash)
    return index.as_query_engine(similarity_top_k=5,streaming=True)

if __name__ == "__main__":
    from parse_document import parse_pdf

    file_path = "uploads/RAG_target_file.pdf"
    docs, document_hash = parse_pdf(file_path, chunk_size=1000, overlap=100)

    # Build the index with Chroma
    index = build_chroma_index(docs, document_hash)