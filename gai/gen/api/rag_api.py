# resolve dir conflict with rag-gen
import sys,os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(this_dir, "../../.."))

from fastapi import FastAPI, Body, HTTPException,Form,File,UploadFile
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import asyncio
import os,json,io
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_lib installed = {dependencies.LIB_VERSION}")
swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
    )
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()

from gai.gen.rag.RAG import RAG
rag = RAG.GetInstance()
# Pre-load embedding model
rag.load()


### ----------------- INDEXING ----------------- ###

class IndexRequest(BaseModel):
    collection_name: str
    text: str
    path_or_url: str
    chunk_size: int = 2000
    chunk_overlap: int = 200
    class Config:
        extra = 'allow'  # Allow extra fields

@app.post("/gen/v1/rag/index")
async def index(request: IndexRequest = Body(...)):
    logger.info(f"main.index: collection_name={request.collection_name}")
    metadata = request.model_dump(exclude={"text", "collection_name", "path_or_url","chunk_size","chunk_overlap"})  # metadata = convert to dict - non-metadata fields
    return rag.index(collection_name=request.collection_name,
        text=request.text, 
        path_or_url=request.path_or_url,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        **metadata)

class IndexChunkRequest(BaseModel):
    collection_name: str
    chunk: str
    path_or_url: str
    class Config:
        extra = 'allow'  # Allow extra fields

@app.post("/gen/v1/rag/index_chunk")
async def index_chunk(request: IndexChunkRequest = Body(...)):
    metadata = request.model_dump(exclude={"chunk", "collection_name", "path_or_url"})  # metadata = convert to dict - non-metadata fields
    return rag.index_chunk(collection_name=request.collection_name,
        chunk=request.chunk, 
        path_or_url=request.path_or_url,
        **metadata)


@app.post("/gen/v1/rag/index_file")
async def index_file(collection_name: str=Form(...), file: UploadFile=File(...), metadata: str=Form(...)):
    text = await file.read()
    text = text.decode("utf-8")
    metadata_dict = json.loads(metadata)
    return rag.index(collection_name=collection_name,
        text=text, 
        path_or_url=file.filename,
        metadata=metadata_dict)

### ----------------- RETRIEVAL ----------------- ###

class QueryRequest(BaseModel):
    collection_name: str
    query_texts: str    
    n_results: int = 3

# POST /gen/v1/rag/retrieve
@app.post("/gen/v1/rag/retrieve")
async def retrieve(request: QueryRequest = Body(...)):
    logger.info(f"main.retrieve: collection_name={request.collection_name}")
    result = rag.retrieve(collection_name=request.collection_name,query_texts=request.query_texts, n_results=request.n_results)
    logger.debug("main.retrieve=",result)
    return result

# GET /gen/v1/rag/retrieve?collection_name
@app.get("/gen/v1/rag/retrieve/{collection_name}/{id}")
def get_document_by_id(self,collection_name, id):
    collection = self._get_collection(collection_name)
    return collection.get(ids=[id])

### ----------------- COLLECTIONS ----------------- ###

# DELETE /gen/v1/rag/delete_collection
@app.delete("/gen/v1/rag/collection/{collection_name}")
async def delete_collection(collection_name):
    rag.delete_collection(collection_name=collection_name)

@app.get("/gen/v1/rag/collection/{collection_name}/count")
async def get_collection_count(collection_name):
    return rag.get_collection_count(collection_name)

@app.get("/gen/v1/rag/collection/{collection_name}")
async def get_collection(collection_name):
    col=rag._get_collection(collection_name)
    return col.get()

@app.put("/gen/v1/rag/collection/{collection_name}")
async def create_collection(collection_name):
    # vs.get_collection is equivalent to get or create so its idempotent
    try:
        rag._get_collection(collection_name)
    except:
        raise HTTPException(status_code=500,detail=f"Failed to create collection {collection_name}.")

@app.get("/gen/v1/rag/collections")
async def list_collections():
    return rag.list_collections()


# REPOSITORIES --------------------------------------------------------------------------------------------------------

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gai.gen.rag.repositories.UserRepository import UserProfileRepository
from gai.gen.rag.repositories.UserDocumentRepository import UserDocumentRepository
engine = create_engine(os.environ["SQLALCHEMY_DATABASE_URI"])
Session = sessionmaker(bind=engine)

@app.get("/gen/v1/rag/repositories/user/{user_profile_id}")
async def user_repository_get_user_by_profile_id(user_profile_id):
    session = Session()    
    repository = UserProfileRepository(session)
    return repository.get_user_by_id(user_profile_id)

@app.get("/gen/v1/rag/repositories/user_documents/{user_profile_id}")
async def user_repository_list_user_documents_by_profile_id(user_profile_id):
    session = Session()    
    repository = UserDocumentRepository(session)
    documents = repository.list_user_documents(user_profile_id)
    session.close()
    return {'documents': [doc.to_dict() for doc in documents]}

# -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12031)
