from datetime import datetime
import uuid
from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, Boolean, create_engine,text)
from sqlalchemy.orm import sessionmaker,joinedload
from gai.gen.rag.models.Base import Base
from gai.gen.rag.models.IndexedDocument import IndexedDocument
from gai.gen.rag.models.IndexedDocumentChunk import IndexedDocumentChunk
from gai.common.logging import logging
logger = logging.getLogger(__name__)

class Repository:

    def __init__(self, conn_string = None):
        if conn_string is None:
            conn_string = 'sqlite:///:memory:'
            logger.info("No connection string provided, using in-memory database")
        try:
            self.engine = create_engine(conn_string)
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.error(f'Repository: Error creating database: conn_string={conn_string}. ' + str(e))
            raise e

    def create_document(self, document, chunk_ids):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            document.Id = str(uuid.uuid4())
            document.ChunkCount = len(chunk_ids)
            document.CreatedAt = datetime.now()
            document.UpdatedAt = datetime.now()
            if document.PublishedDate and isinstance(document.PublishedDate,str):
                try:
                    document.PublishedDate = datetime.strptime(document.PublishedDate, '%Y-%b-%d')
                except:
                    document.PublishedDate = None
            else:
                document.PublishedDate = None

            for id in chunk_ids:
                document_chunk = IndexedDocumentChunk(
                    Id=id,
                    ChunkId = id,
                    DocumentId=document.Id,
                )
                document.chunks.append(document_chunk)
            session.add(document)
            session.commit()
            return document.Id
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def get_document(self, document_id):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            # Use joinedload to "include" the chunks when loading a document
            return session.query(IndexedDocument).options(joinedload(IndexedDocument.chunks)).filter_by(Id=document_id).first()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def get_document_from_chunk_id(self, chunk_id):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            return session.query(IndexedDocumentChunk).filter_by(Id=chunk_id).first().DocumentId
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def list_documents(self, collection_name):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            return session.query(IndexedDocument).filter_by(CollectionName=collection_name).all()
        except:
            session.rollback()
            raise
        finally:
            session.close() 

    def list_docids(self, collection_name):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            return [doc[0] for doc in session.query(IndexedDocument.Id).filter(IndexedDocument.CollectionName==collection_name).all()]
        except:
            session.rollback()
            raise
        finally:
            session.close() 

    def delete_document(self, document_id):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            sql = text("DELETE FROM IndexedDocumentChunks WHERE DocumentId = :id")
            session.execute(sql, {"id": document_id})
            session.commit()

            sql = text("DELETE FROM IndexedDocuments WHERE id = :id")
            session.execute(sql, {"id": document_id})
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def update_document(self, document):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            existing_doc = session.query(IndexedDocument).filter_by(Id=document.Id).first()

            if existing_doc is not None:
                # Update all fields as necessary
                existing_doc.FileName = document.FileName
                existing_doc.Source = document.Source
                existing_doc.Abstract = document.Abstract
                existing_doc.Authors = document.Authors
                existing_doc.Title = document.Title
                existing_doc.Publisher = document.Publisher

                if document.PublishedDate and isinstance(document.PublishedDate,str):
                    try:
                        existing_doc.PublishedDate = datetime.strptime(document.PublishedDate, '%Y-%b-%d')
                    except:
                        existing_doc.PublishedDate = None
                else:
                    existing_doc.PublishedDate = None

                existing_doc.Comments = document.Comments
                existing_doc.UpdatedAt = datetime.now()

                session.commit()
                return existing_doc
            else:
                raise ValueError("No document found with the provided Id.")
        except:
            session.rollback()
            raise
        finally:
            session.close()