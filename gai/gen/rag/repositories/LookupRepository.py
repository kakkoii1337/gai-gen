# resolve dir conflict with rag-gen
import sys,os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(this_dir, "../../.."))

from gai.rag.models.DocumentType import DocumentType
from sqlalchemy.orm import Session

class LookupRepository:

    def __init__(self, session: Session):
        self.session = session

    def list_document_types(self):
        return self.session.query(DocumentType).all()
