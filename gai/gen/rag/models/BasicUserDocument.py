from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BasicUserDocument(Base):
    __tablename__ = 'BasicUserDocuments'

    Id =                    Column(String(36), primary_key=True)
    DocumentTypeId =        Column(String(36), ForeignKey('DocumentTypes.Id'), nullable=False)
    BasicUserProfilesId =   Column(String(36), ForeignKey('BasicUserProfiles.Id'), nullable=False)
    FileName =              Column(String(200), nullable=False)
    File =                  Column(LONGBLOB, nullable=False)
    IsActive =              Column(Boolean, default=True)
    CreatedAt =             Column(DateTime)
    UpdatedAt =             Column(DateTime)

    # Relationships
    document_type = relationship('DocumentType', back_populates='basic_user_documents')
    basic_user_profile = relationship('BasicUserProfile', back_populates='basic_user_documents')
