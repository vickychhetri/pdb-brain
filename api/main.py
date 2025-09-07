import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from docling.document_converter import DocumentConverter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

from fastapi.responses import StreamingResponse
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import asyncio

from fastapi.middleware.cors import CORSMiddleware



# -------------------
# Init services
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
qdrant = QdrantClient(url="http://qdrant:6333")
COLLECTION_NAME = "documents"

# Embeddings from Ollama (Mistral)
embeddings = OllamaEmbeddings(model="mistral", base_url="http://ollama:11434")

# LLM (Mistral via Ollama)
llm = OllamaLLM(model="mistral", base_url="http://ollama:11434")

# Set vector size manually (Mistral embeddings = 4096 dims)
VECTOR_SIZE = 4096

# Ensure collection exists
collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE
        ),
    )

# -------------------
# API Routes
# -------------------

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into Qdrant after parsing with Docling."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert with Docling
    converter = DocumentConverter()
    result = converter.convert(tmp_path)
    text = result.document.export_to_markdown()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Store in Qdrant
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_texts(chunks)

    return {"status": "success", "chunks_ingested": len(chunks)}

@app.get("/query")
async def query_document(q: str):
    """Query ingested documents and generate an answer with Mistral (RAG)."""
    vector_store = QdrantVectorStore(
        client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Build a RetrievalQA chain (retriever + Mistral LLM)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Stuff = concatenate retrieved docs into context
        return_source_documents=True,
    )

    response = qa_chain.invoke({"query": q})

    return {
        "query": q,
        "answer": response["result"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }


# Custom callback handler for streaming tokens
class StreamHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        await self.queue.put("[[END]]")  # signal end


@app.get("/query_stream")
async def query_document_stream(q: str):
    """Stream Mistral's answer in real-time (RAG)."""
    vector_store = QdrantVectorStore(
        client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Streaming queue
    queue: asyncio.Queue = asyncio.Queue()
    handler = StreamHandler(queue)

    # Create a streaming LLM
    streaming_llm = OllamaLLM(
        model="mistral",
        base_url="http://ollama:11434",
        streaming=True,
        callbacks=[handler],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=streaming_llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )

    # Run the chain in background (so it pushes tokens into queue)
    async def run_chain():
        await asyncio.to_thread(qa_chain.invoke, {"query": q})

    asyncio.create_task(run_chain())

    # Stream tokens as SSE
    async def token_generator():
        while True:
            token = await queue.get()
            if token == "[[END]]":
                yield "data: [DONE]\n\n"
                break
            # SSE requires "data: ...\n\n"
            yield f"data: {token}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# # Custom callback handler for streaming tokens
# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, queue: asyncio.Queue):
#         self.queue = queue

#     async def on_llm_new_token(self, token: str, **kwargs) -> None:
#         await self.queue.put(token)

#     async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
#         await self.queue.put("[[END]]")  # signal end

# @app.get("/query_stream")
# async def query_document_stream(q: str):
#     """Stream Mistral's answer in real-time (RAG)."""
#     vector_store = QdrantVectorStore(
#         client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
#     )
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#     # Streaming queue
#     queue: asyncio.Queue = asyncio.Queue()
#     handler = StreamHandler(queue)

#     # Create a streaming LLM
#     streaming_llm = OllamaLLM(
#         model="mistral",
#         base_url="http://ollama:11434",
#         streaming=True,
#         callbacks=[handler],
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=streaming_llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=False,  # sources can be added separately if needed
#     )

#     # Run the chain in background
#     async def run_chain():
#         await asyncio.to_thread(qa_chain.invoke, {"query": q})

#     asyncio.create_task(run_chain())

#     # Stream tokens as they arrive
#     async def token_generator():
#         while True:
#             token = await queue.get()
#             if token == "[[END]]":
#                 break
#             yield token

#     return StreamingResponse(token_generator(), media_type="text/plain")