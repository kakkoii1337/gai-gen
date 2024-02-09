from sqlalchemy import Column, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date,BIGINT
from sqlalchemy.orm import relationship
from gai.gen.rag.models.Base import Base
from gai.gen.rag.models.IndexedDocumentChunk import IndexedDocumentChunk

class IndexedDocument(Base):
    __tablename__ = 'IndexedDocuments'

    Id = Column(VARCHAR(36), primary_key=True)
    CollectionName = Column(VARCHAR(200), nullable=False)
    ChunkCount = Column(INTEGER, nullable=False)
    ByteSize = Column(BIGINT, nullable=False)
    ChunkSize = Column(INTEGER, nullable=False)
    Overlap = Column(INTEGER, nullable=False)
    SplitAlgo = Column(VARCHAR(200))
    FileName = Column(VARCHAR(200))
    Source = Column(VARCHAR(255))
    Abstract = Column(Text)
    Authors = Column(VARCHAR(255))
    Title = Column(VARCHAR(255))
    Publisher = Column(VARCHAR(255))
    PublishedDate = Column(Date)
    Comments = Column(Text)
    IsActive = Column(Boolean, default=True)
    CreatedAt = Column(DateTime)
    UpdatedAt = Column(DateTime)

    # One-to-many relationship with IndexedDocumentChunk
    chunks = relationship("IndexedDocumentChunk", backref="document")