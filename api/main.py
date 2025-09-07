import os
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


# -------------------
# Init FastAPI
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Init Services
# -------------------
qdrant = QdrantClient(url="http://qdrant:6333")
COLLECTION_NAME = "documents"

# Embeddings from Ollama (Mistral)
embeddings = OllamaEmbeddings(model="mistral", base_url="http://ollama:11434")

# LLM (non-streaming)
llm = OllamaLLM(model="mistral", base_url="http://ollama:11434")

# Vector size (Mistral embeddings = 4096 dims)
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

# Directory to store uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------
# Ingest Endpoint
# -------------------
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document into Qdrant."""
    saved_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # Convert to text
    converter = DocumentConverter()
    result = converter.convert(saved_path)
    text = result.document.export_to_markdown()

    # Detect document category
    prompt = f"""
    Classify the following document into one of these categories:
    Agreement, Legal Notice, Court Document, Invoice, Bank Statement, Tax Return, Audit Report, Payroll, HR Document, Report, Proposal, Insurance, Certificate, Other

    Document text: {text[:2000]}
    Respond with only the category name.
    """
    doc_category = llm.predict(prompt).strip() or "Other"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)

    # Store chunks in Qdrant
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_texts(
        texts=chunks,
        metadatas=[{"filename": file.filename, "doctype": doc_category} for _ in chunks]
    )

    return {
        "status": "success",
        "chunks_ingested": len(chunks),
        "filename": file.filename,
        "category": doc_category
    }


# -------------------
# Normal Query Endpoint
# -------------------
@app.get("/query")
async def query_document(q: str):
    """Query documents with RAG (no streaming)."""
    vector_store = QdrantVectorStore(
        client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    response = qa_chain.invoke({"query": q})

    return {
        "query": q,
        "answer": response["result"],
        "sources": [
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename"),
                "doctype": doc.metadata.get("doctype")
            }
            for doc in response["source_documents"]
        ]
    }


# -------------------
# Streaming Handler
# -------------------
class StreamHandler(BaseCallbackHandler):
    """
    Custom handler to capture streaming tokens from the LLM.
    Each new token is added into a queue.
    """
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs):
        # Every time the model generates a token, put it in the queue
        await self.queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs):
        # When the model finishes generating, put an END signal
        await self.queue.put("[[END]]")


# -------------------
#  Streaming Query Endpoint (real-time SSE)
# -------------------

#  What’s happening here:
# /query endpoint:
# Works in blocking mode (waits until the model finishes).
# Fetches documents from Qdrant, sends them to Mistral, and returns the full answer with sources.

# StreamHandler:
# Captures tokens as the LLM generates them.
# Stores them in an asyncio.Queue so they can be consumed later.

# /query_stream endpoint:
# Uses SSE (Server-Sent Events) to stream responses token by token.
# Sends partial output in real time to the frontend.
# At the end, also sends sources (which documents were used).
# Frontend behavior:
# If you connect with JavaScript EventSource to /query_stream, you’ll see tokens appear live as they’re generated.

# - Vicky Chhetri
@app.get("/query_stream")
async def query_document_stream(q: str):
    vector_store = QdrantVectorStore(
        client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    queue: asyncio.Queue = asyncio.Queue()
    handler = StreamHandler(queue)

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
        return_source_documents=True,
    )

    async def run_chain():
        result = await qa_chain.ainvoke({"query": q})

        sources = [
            {
                "filename": doc.metadata.get("filename"),
                "doctype": doc.metadata.get("doctype"),
                "preview": doc.page_content[:200],
            }
            for doc in result["source_documents"]
        ]
        
        await queue.put({"type": "sources", "data": sources})
        print(">>> sending sources:", sources)
        await queue.put("[[END]]")

    asyncio.create_task(run_chain())

    async def token_generator():
        while True:
            item = await queue.get()
            print(item)
            if item == "[[END]]":
                yield "data: [DONE]\n\n"
                break
            elif isinstance(item, dict) and item.get("type") == "sources":
                print(f"event: sources\ndata: {json.dumps(item['data'])}\n\n")
                yield f"event: sources\ndata: {json.dumps(item['data'])}\n\n"
            else:
                yield f"data: {item}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# @app.get("/query_stream")
# async def query_document_stream(q: str):
#     """Stream answer from Mistral (SSE)"""
#     """
#     Stream answer from Mistral using Server-Sent Events (SSE).
#     Instead of waiting for the full answer, tokens are sent one by one.
#     """
#     # Connect to Qdrant again
#     vector_store = QdrantVectorStore(
#         client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings
#     )
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#     # Create an async queue to hold generated tokens
#     queue: asyncio.Queue = asyncio.Queue()
    
#     # Pass this queue to our custom stream handler
#     handler = StreamHandler(queue)

#     # Create a streaming LLM (Ollama Mistral)
#     streaming_llm = OllamaLLM(
#         model="mistral",                 # Which LLM to use
#         base_url="http://ollama:11434",  # Ollama endpoint
#         streaming=True,                  # Enable streaming mode
#         callbacks=[handler],             # Use our StreamHandler to capture tokens
#     )

#     # Build a RetrievalQA chain with the streaming LLM
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=streaming_llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=True,
#     )

#     # This async function runs in the background to generate results
#     async def run_chain():
#         result = qa_chain.invoke({"query": q})

#         # Send sources after completion : After full generation is done, send the sources too
#         sources = [
#             {
#                 "filename": doc.metadata.get("filename"),
#                 "doctype": doc.metadata.get("doctype"),
#                 "preview": doc.page_content[:200],
#             }
#             for doc in result["source_documents"]
#         ]

#         # Send sources to the queue
#         await queue.put({"type": "sources", "data": sources})
        
#         # Signal the end
#         await queue.put("[[END]]")

#     # Run the chain asynchronously without blocking
#     asyncio.create_task(run_chain())

#     # Generator that yields SSE messages as tokens come in
#     async def token_generator():
#         while True:
#             item = await queue.get()
#             if item == "[[END]]":
#                  # End of stream
#                 yield "data: [DONE]\n\n"
#                 break
#             elif isinstance(item, dict) and item.get("type") == "sources":
#                  # Send sources as a separate SSE event
#                 yield f"event: sources\ndata: {json.dumps(item['data'])}\n\n"
#             else:
#                 # Send each token as it arrives
#                 yield f"data: {item}\n\n"
#     # Return a StreamingResponse with SSE format
#     return StreamingResponse(token_generator(), media_type="text/event-stream")


# -------------------
# File Download
# -------------------
@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download an uploaded file."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)
