# resolve dir conflict with rag-gen
import sys,os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(this_dir, "../../.."))
from gai.gen.rag.models.BasicUserDocument import BasicUserDocument
from sqlalchemy.orm import Session

class UserDocumentRepository:

    def __init__(self, session: Session):
        self.session = session

    def list_user_documents(self, user_id):
        session = self.Session()
        documents = session.query(BasicUserDocument).filter_by(user_id=user_id).all()
        session.close()
        return documents

    def get_document_by_id(self, document_id):
        session = self.Session()
        document = session.query(BasicUserDocument).get(document_id)
        session.close()
        return document

    def add_document(self, document):
        session = self.Session()
        session.add(document)
        session.commit()
        session.close()

    def update_document(self, document):
        session = self.Session()
        session.merge(document)
        session.commit()
        session.close()

    def delete_document(self, document_id):
        session = self.Session()
        document = session.query(BasicUserDocument).get(document_id)
        if document is not None:
            session.delete(document)
            session.commit()
        session.close()