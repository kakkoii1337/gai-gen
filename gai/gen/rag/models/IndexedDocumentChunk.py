from sqlalchemy import Column, ForeignKey, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date
from gai.gen.rag.models.Base import Base

class IndexedDocumentChunk(Base):
    __tablename__ = 'IndexedDocumentChunks'

    Id = Column(VARCHAR(36), primary_key=True)
    ChunkId = Column(VARCHAR(64))
    DocumentId = Column(VARCHAR(36), ForeignKey('IndexedDocuments.Id'))

