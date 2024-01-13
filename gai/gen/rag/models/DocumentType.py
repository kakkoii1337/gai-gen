from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentType(Base):
    __tablename__ = 'DocumentTypes'

    Id = Column(String(36), primary_key=True)
    Type = Column(String(50), nullable=False, unique=True)
    IsActive = Column(Boolean, default=True)
    UpdatedAt = Column(DateTime)
    CreatedAt = Column(DateTime)

    # Relationships
    basic_user_documents = relationship('BasicUserDocuments', back_populates='document_type')
