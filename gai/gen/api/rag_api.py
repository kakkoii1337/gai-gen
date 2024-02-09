from gai.gen.api.globals import status_updater
from gai.gen.api.status_update_router import status_update_router
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from gai.gen.rag import RAG
from gai.gen import Gaigen
from gai.common.logging import getLogger
import dependencies
from gai.common.PDFConvert import PDFConvert
import tempfile
from fastapi import FastAPI, Body, Form, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import json
from gai.gen.api.errors import *
load_dotenv()


# Configure Dependencies
dependencies.configure_logging()
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_lib installed = {dependencies.LIB_VERSION}")
swagger_url = dependencies.get_swagger_url()
app = FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
)
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()

# Add status update router
app.include_router(status_update_router)

gen = Gaigen.GetInstance()

# Pre-load default model


def preload_model():
    try:
        # RAG does not use "default" model
        gen.load("rag")
    except Exception as e:
        logger.error(f"Failed to preload default model: {e}")
        raise e


preload_model()

# RAG specific

### ----------------- INDEXING ----------------- ###


class IndexRequest(BaseModel):
    collection_name: str
    text: str
    path_or_url: str
    chunk_size: int = 2000
    chunk_overlap: int = 200

    class Config:
        extra = 'allow'  # Allow extra fields


# @app.post("/gen/v1/rag/index")
# async def index(request: IndexRequest = Body(...)):
#     try:
#         logger.info(f"main.index: collection_name={request.collection_name}")
#         # metadata = convert to dict - non-metadata fields
#         metadata = request.model_dump(
#             exclude={"text", "collection_name", "path_or_url", "chunk_size", "chunk_overlap"})
#         return gen.index(collection_name=request.collection_name,
#                          text=request.text,
#                          path_or_url=request.path_or_url,
#                          chunk_size=request.chunk_size,
#                          chunk_overlap=request.chunk_overlap,
#                          **metadata)
#     except Exception as e:
#         return InternalError(str(e))

@app.post("/gen/v1/rag/index-file")
async def index_file(collection_name: str = Form(...), file: UploadFile = File(...), metadata: str = Form(...)):
    try:
        # We will use a simple file extension match to determine if the file is PDF.
        # This is not a robust way to determine file type, but it is sufficient for our purposes.
        if file.filename.endswith(".pdf"):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open(temp_file.name, "wb") as f:
                f.write(await file.read())
            text = PDFConvert.pdf_to_text(temp_file.name)

        else:
            text = await file.read()
            text = text.decode("utf-8")
        metadata_dict = json.loads(metadata)

        return await gen.index_async(collection_name=collection_name,
                                     text=text,
                                     path_or_url=file.filename,
                                     metadata=metadata_dict,
                                     status_updater=status_updater)
    except Exception as e:
        return InternalError(str(e))

### ----------------- RETRIEVAL ----------------- ###

class QueryRequest(BaseModel):
    collection_name: str
    query_texts: str
    n_results: int = 3

# Retrieve document chunks using semantic search
# POST /gen/v1/rag/retrieve


@app.post("/gen/v1/rag/retrieve")
async def retrieve(request: QueryRequest = Body(...)):
    try:
        logger.info(
            f"main.retrieve: collection_name={request.collection_name}")
        result = gen.retrieve(collection_name=request.collection_name,
                              query_texts=request.query_texts, n_results=request.n_results)
        logger.debug("main.retrieve=", result)
        return result
    except Exception as e:
        return InternalError(str(e))

# Get a document by its ID
# GET /gen/v1/rag/retrieve?collection_name


@app.get("/gen/v1/rag/retrieve/{collection_name}/{id}")
def get_document_by_id(self, collection_name, id):
    try:
        collection = self._get_collection(collection_name)
        return collection.get(ids=[id])
    except Exception as e:
        return InternalError(str(e))

### ----------------- COLLECTIONS ----------------- ###

# DELETE /gen/v1/rag/collection/{}
@app.delete("/gen/v1/rag/collection/{collection_name}")
async def delete_collection(collection_name):
    try:
        RAG.delete_collection(collection_name=collection_name)
        after = RAG.list_collections()
        return { "count": after.count }
    except Exception as e:
        return InternalError(str(e))

# GET /gen/v1/rag/collections
@app.get("/gen/v1/rag/collections")
async def list_collections():
    try:
        return [collection.name for collection in RAG.list_collections()]
    except Exception as e:
        return InternalError(str(e))

# GET /gen/v1/rag/collection/{collection_name}
@app.get("/gen/v1/rag/collection/{collection_name}")
async def list_documents(collection_name):
    try:
        docs = RAG.list_documents(collection_name=collection_name)
        return [{"id":doc.Id,"title":doc.Title,"size":doc.ByteSize,"chunk_count":doc.ChunkCount,"chunk_size":doc.ChunkSize,"overlap_size":doc.Overlap,"source":doc.Source} for doc in docs]
    except Exception as e:
        return InternalError(str(e))

@app.get("/gen/v1/rag/document/{document_id}")
async def get_document(document_id):
    try:
        return RAG.get_document(document_id=document_id)
    except Exception as e:
        return InternalError(str(e))

class UpdateDocumentRequest(BaseModel):
    Id: str
    FileName: str = None
    Source: str = None
    Abstract: str = None
    Authors: str = None
    Title: str = None
    Publisher: str = None
    PublishedDate: str = None
    Comments: str = None
@app.post("/gen/v1/rag/document")
async def update_document(req: UpdateDocumentRequest = Body(...)):
    try:
        #req_dict = req.model_dump()
        return RAG.update_document(document=req)
    except Exception as e:
        return InternalError(str(e))

@app.delete("/gen/v1/rag/document/{document_id}")
async def delete_document(document_id):
    try:
        return RAG.delete_document(document_id=document_id)
    except Exception as e:
        return InternalError(str(e))

# -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12031)
