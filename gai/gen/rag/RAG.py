import time
import os
import torch
import gc
from tqdm import tqdm
from datetime import datetime
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction
from chromadb.config import Settings
import chromadb
from gai.common.utils import get_config, get_config_path
import threading
from gai.common import logging, file_utils
from gai.common.StatusUpdater import StatusUpdater
from gai.gen.rag.Repository import Repository
from gai.gen.rag.models.IndexedDocument import IndexedDocument
logger = logging.getLogger(__name__)


class RAG:

    def __init__(self, status_updater=None):
        self.generator_name = "rag"
        config = get_config()["gen"]["rag"]
        app_path = get_config_path()
        self.model_path = os.path.join(app_path, config["model_path"])
        if (os.environ.get("RAG_MODEL_PATH")):
            model_path = os.environ["RAG_MODEL_PATH"]
            self.model_path = os.path.join(app_path, model_path)
        self.chromadb_path = os.path.join(app_path, config["chromadb"]["path"])
        
        # sql conn
        self.sqlite_path = os.path.join(app_path, config["sqlite"]["path"])
        self.sqlite_string = f'sqlite:///{self.sqlite_path}'
        self.repo = Repository(self.sqlite_string)

        self.n_results = config["chromadb"]["n_results"]
        self.chunks_path = os.path.join(app_path, config["chunks"]["path"])
        self.chunk_size = config["chunks"]["size"]
        self.chunk_overlap = config["chunks"]["overlap"]
        self.device = config["device"]
        # for thread safety, using Semaphore allows for easier upgrade to support multiple generators in the future
        self.semaphore = threading.Semaphore(1)
        self._db = chromadb.PersistentClient(
            path=self.chromadb_path, settings=Settings(allow_reset=True))
        self.status_updater = status_updater

    # This is idempotent
    def load(self):
        self._ef = InstructorEmbeddingFunction(
            self.model_path, device=self.device)

    def unload(self):
        try:
            del self._ef
            del self._db
        except:
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

    # COLLECTIONS
    # get from chromadb
    @staticmethod
    def get_db():
        try:
            app_path = get_config_path()
            config = get_config()["gen"]["rag"]
            chromadb_path = os.path.join(app_path, config["chromadb"]["path"])
            db = chromadb.PersistentClient(
                path=chromadb_path, settings=Settings(allow_reset=True))
            return db
        except Exception as e:
            if not "does not exist." in str(e):
                raise e

    # get from chromadb
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

    # get from chromadb
    @staticmethod
    def delete_collection(collection_name):
        logger.info(f"Deleting {collection_name}...")
        db = RAG.get_db()
        if RAG.collection_exists(collection_name):
            db.delete_collection(collection_name)
            
        repo = RAG.get_repo()
        docids = repo.list_docids(collection_name)
        for docid in docids:
            repo.delete_document(docid)
    
    # get from chromadb
    @staticmethod
    def get_collection(collection_name):
        db = RAG.get_db()
        return db.get_or_create_collection(collection_name)

    # get from chromadb
    @staticmethod
    def create_collection(collection_name):
        db = RAG.get_db()
        return db.get_or_create_collection(collection_name)

    # get from chromadb
    @staticmethod
    def list_collections():
        db = RAG.get_db()
        return db.list_collections()

    # get from chromadb
    @staticmethod
    def list_chunks(collection_name):
        return RAG.get_collection(collection_name).get()

    # get from chromadb
    @staticmethod
    def get_chunk(collection_name, id):
        collection = RAG.get_collection(collection_name)
        return collection.get(ids=[id])

    # get from sqlite
    @staticmethod
    def get_repo():
        app_path = get_config_path()
        config = get_config()["gen"]["rag"]
        sqlite_path = os.path.join(app_path, config["sqlite"]["path"])
        sqlite_string = f'sqlite:///{sqlite_path}'
        return Repository(sqlite_string)

    # get from sqlite
    @staticmethod
    def list_documents(collection_name):
        repo = RAG.get_repo()
        collection = repo.list_documents(collection_name)
        return collection

    # get from sqlite
    @staticmethod
    def get_document(document_id):
        repo = RAG.get_repo()
        document = repo.get_document(document_id)
        return document

    # get from sqlite
    @staticmethod
    def update_document(document):
        repo = RAG.get_repo()
        return repo.update_document(document)

    # get from sqlite
    @staticmethod
    def delete_document(document_id):
        repo = RAG.get_repo()
        vs = RAG.get_db()

        # delete from vector store
        doc = repo.get_document(document_id)
        collection_name = doc.CollectionName
        collection = vs.get_collection(collection_name)
        collection.delete(ids=[chunk.ChunkId for chunk in doc.chunks])

        # delete from sqlite
        repo.delete_document(document_id)
        return repo.get_document(document_id)

    # INDEXING
    def _get_collection(self, collection_name):
        try:
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
    def index_chunk(self, collection_name, chunk, path_or_url, metadata={"source": "unknown"}):
        try:
            chunks_dir = file_utils.get_chunk_dir(
                self.chunks_path, path_or_url)
            curr_time = time.time()
            utc = datetime.utcfromtimestamp(curr_time)
            time_s = utc.strftime('%Y-%m-%d %H:%M:%S')
            data = {
                "chunks_dir": chunks_dir,
                "created": time_s
            }
            if metadata:
                data = {**data, **metadata}
            collection = self._get_collection(collection_name)
            chunk_id = file_utils.create_chunk_id(chunk)
            collection.upsert(documents=[chunk], metadatas=[
                              data], ids=[chunk_id])
            return chunk_id
        except Exception as error:
            logger.error(f"vector_store.index_chunk: error={error}")
            raise error

    # Split text in temp dir and index each chunk into vector store locally
    async def index_async(self, collection_name, text, path_or_url, metadata={"source": "unknown"}, chunk_size=None, chunk_overlap=None, status_updater=None):
        chunks = []
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size
            if chunk_overlap is None:
                chunk_overlap = self.chunk_overlap
            chunks_dir = file_utils.get_chunk_dir(
                self.chunks_path, path_or_url)
            file_utils.split_chunks(text=text,
                                    chunks_dir=chunks_dir,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap)
            chunks = os.listdir(chunks_dir)
        except Exception as error:
            logger.error(
                f"vector_store.index: Failed to split chunks. error={error}")
            raise error
        ids = []
        count = len(chunks)
        for i, chunk_id in tqdm(enumerate(chunks)):
            with open(os.path.join(chunks_dir, chunk_id), 'r') as f:
                chunk = f.read()
            self.index_chunk(collection_name, chunk, chunks_dir, metadata)
            ids.append(chunk_id)
            logger.debug(
                f"Indexed {i}/{count} chunk {chunk_id} into collection {collection_name}")

            # Callback for progress update
            if status_updater:
                await status_updater.update_progress(i, len(chunks))

        # Update sqlite
        try:
            doc = IndexedDocument()
            doc.CollectionName = collection_name
            doc.ChunkSize = chunk_size
            doc.ByteSize = len(text)
            doc.Overlap = chunk_overlap
            doc.Title = metadata.get('title', '')
            doc.FileName = metadata.get('filename', '')
            doc.Source = metadata.get('source', '')
            doc.Authors = metadata.get('authors', '')
            doc.Abstract = metadata.get('abstract', '')
            doc.PublishedDate = metadata.get('published_date', '')
            doc.Comments = metadata.get('comments', '')
            doc.CreatedAt = datetime.now()
            doc.UpdatedAt = datetime.now()
            doc.IsActive = True

            doc_id = self.repo.create_document(doc, ids)
            logger.debug(f"Indexed document {doc_id} into sqlite")

            return doc_id
        except Exception as error:
            logger.error(f"vector_store.index: Failed to insert document into sqlite. error={error}")
            raise error

    # RETRIEVAL

    def retrieve(self, collection_name, query_texts, n_results=None):
        logger.info(f"Retrieving by query {query_texts}...")

        collection = self._get_collection(collection_name)
        if n_results is None:
            n_results = self.n_results
        result = collection.query(query_texts=query_texts, n_results=n_results)

        # Not found
        if 'ids' not in result or result['ids'] is None or len(result['ids']) == 0 or len(result['ids'][0]) == 0:
            return None

        if len(result['ids']) > 0:
            logger.debug('result=', result)

        import pandas as pd
        df = pd.DataFrame({
            'documents': result['documents'][0],
            'metadatas': result['metadatas'][0],
            'distances': result['distances'][0],
            'ids': result['ids'][0]
        })

        # drop duplicates
        return df.drop_duplicates(subset=['ids']).sort_values('distances', ascending=True)

