from sqlalchemy import Column, DateTime, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BasicUserProfile(Base):
    __tablename__ = 'BasicUserProfiles'
    
    Id =            Column(String(36), primary_key=True)
    UserName =      Column(String(128), nullable=False)
    UserEmail =     Column(String(255), nullable=False, unique=True)
    IsActive =      Column(Boolean, nullable=False, default=True)

    documents = relationship("BasicUserDocument", back_populates="user")