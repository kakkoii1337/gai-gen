import threading
from gai.common import logging,file_utils
logger = logging.getLogger(__name__)

from gai.common.utils import get_config, get_config_path
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction

import gc, torch, os, time 
from datetime import datetime
from tqdm import tqdm

class RAG:

    def __init__(self):
        self.generator_name="rag"
        config = get_config()["gen"]["rag"]
        app_path = get_config_path()
        self.model_path =  os.path.join(app_path, config["model_path"])
        if (os.environ.get("RAG_MODEL_PATH")):
            model_path = os.environ["RAG_MODEL_PATH"]
            self.model_path = os.path.join(app_path, model_path)
        self.chromadb_path = os.path.join(app_path, config["chromadb"]["path"])
        self.n_results = config["chromadb"]["n_results"]
        self.chunks_path = os.path.join(app_path, config["chunks"]["path"])
        self.chunk_size = config["chunks"]["size"]
        self.chunk_overlap = config["chunks"]["overlap"]
        self.device = config["device"]
        self.semaphore = threading.Semaphore(1)     # for thread safety, using Semaphore allows for easier upgrade to support multiple generators in the future
        self._db = chromadb.PersistentClient(path=self.chromadb_path,settings=Settings(allow_reset=True))

    # This is idempotent
    def load(self):
        self._ef = InstructorEmbeddingFunction(self.model_path,device=self.device)
       
    def unload(self):
        try:
            del self._ef
            del self._db
        except :
            pass
        self._db = None
        self._ef = None
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self):
        logger.info("Deleting database...")
        try:
            self._db.reset
        except Exception as e:
            if not "does not exist." in str(e):
                raise e

    ## COLLECTIONS
    @staticmethod
    def get_db():
        try:
            app_path = get_config_path()
            config = get_config()["gen"]["rag"]
            chromadb_path = os.path.join(app_path, config["chromadb"]["path"])
            db = chromadb.PersistentClient(path=chromadb_path,settings=Settings(allow_reset=True))
            return db
        except Exception as e:
            if not "does not exist." in str(e):
                raise e

    @staticmethod
    def collection_exists(collection_name):
        db = RAG.get_db()
        try:
            db.get_collection(collection_name)
            return True
        except Exception as e:
            if "does not exist." in str(e):
                return False
            raise e

    @staticmethod
    def delete_collection(collection_name):
        logger.info(f"Deleting {collection_name}...")
        db = RAG.get_db()
        if RAG.collection_exists(collection_name):
            db.delete_collection(collection_name)

    @staticmethod
    def get_collection(collection_name):
        db = RAG.get_db()
        return db.get_or_create_collection(collection_name)

    @staticmethod
    def create_collection(collection_name):
        db = RAG.get_db()
        return db.get_or_create_collection(collection_name)        

    @staticmethod
    def list_collections():
        db = RAG.get_db()
        return db.list_collections()

    @staticmethod
    def get_document_count(self, collection_name):
        return RAG.get_collection(collection_name).count()

    ## INDEXING
    def _get_collection(self,collection_name):
        try:
            logger.debug(f"RAG._get_collection: collection_name={collection_name}")
            collection = self._db.get_or_create_collection(
                        collection_name,
                        embedding_function=self._ef,
                        metadata={"hnsw:space": "cosine"} 
                    )
        except Exception as error:
            logger.error(f"RAG._get_collection: error={error}")
            raise error
        return collection

    # Index chunk of text into vector store locally
    def index_chunk(self,collection_name,chunk,path_or_url, metadata={"source":"unknown"}):
        try:
            chunks_dir = file_utils.get_chunk_dir(self.chunks_path, path_or_url)
            curr_time = time.time()
            utc = datetime.utcfromtimestamp(curr_time)
            time_s = utc.strftime('%Y-%m-%d %H:%M:%S')
            data = {
                "chunks_dir":chunks_dir,
                "created":time_s
            }
            if metadata:
                data = {**data, **metadata}
            collection = self._get_collection(collection_name)
            chunk_id = file_utils.create_chunk_id(chunk)
            collection.upsert(documents=[chunk], metadatas=[data], ids=[chunk_id])
            return chunk_id
        except Exception as error:
            logger.error(f"vector_store.index_chunk: error={error}")
            raise error

    # Split text in temp dir and index each chunk into vector store locally
    def index(self,collection_name, text, path_or_url, metadata={"source":"unknown"}, chunk_size=None, chunk_overlap=None):
        chunks=[]
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size
            if chunk_overlap is None:
                chunk_overlap = self.chunk_overlap
            chunks_dir = file_utils.get_chunk_dir(self.chunks_path,path_or_url)
            file_utils.split_chunks(text=text, 
                chunks_dir=chunks_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)
            chunks = os.listdir(chunks_dir)
        except Exception as error:
            logger.error(f"vector_store.index: Failed to split chunks. error={error}")
            raise error
        ids=[]
        for chunk_id in tqdm(chunks):
            with open(os.path.join(chunks_dir,chunk_id),'r') as f:
                chunk = f.read()
            self.index_chunk(collection_name,chunk, chunks_dir, metadata)
            ids.append(chunk_id)
        return ids
    
    ## RETRIEVAL

    def retrieve(self, collection_name, query_texts, n_results=None):
        logger.info(f"Retrieving by query {query_texts}...")

        collection = self._get_collection(collection_name)
        if n_results is None:
            n_results = self.n_results
        result = collection.query(query_texts=query_texts,n_results=n_results)

        # Not found
        if 'ids' not in result or result['ids'] is None or len(result['ids'])==0 or len(result['ids'][0])==0:
            return None
       
        if len(result['ids']) > 0:
            logger.debug('result=',result)

        import pandas as pd
        df = pd.DataFrame({
                'documents': result['documents'][0],
                'metadatas': result['metadatas'][0],
                'distances': result['distances'][0],
                'ids': result['ids'][0]
            })

        # drop duplicates
        return df.drop_duplicates(subset=['ids']).sort_values('distances', ascending=True)

    def retrieve_by_id(self,collection_name, id):
        collection = self._get_collection(collection_name)
        return collection.get(ids=[id])


