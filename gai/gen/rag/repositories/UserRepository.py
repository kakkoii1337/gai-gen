# resolve dir conflict with rag-gen
import sys,os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(this_dir, "../../.."))
from gai.gen.rag.models.BasicUserProfile import BasicUserProfile
from sqlalchemy.orm import Session

class UserProfileRepository:

    def __init__(self, session: Session):
        self.session = session

    def list_users(self):
        return self.session.query(BasicUserProfile).all()

    def get_user_by_id(self, user_id: str):
        return self.session.query(BasicUserProfile).filter_by(Id=user_id).first()

    def create_user(self, user: BasicUserProfile):
        self.session.add(user)
        self.session.commit()
        return user

    def update_user(self, user: BasicUserProfile):
        existing_user = self.get_user_by_id(user.Id)
        if existing_user is not None:
            for key, value in vars(user).items():
                if value is not None:
                    setattr(existing_user, key, value)
            self.session.commit()
            return existing_user
        else:
            raise ValueError(f"User with id {user.Id} does not exist.")

    def delete_user(self, user_id: str):
        user = self.get_user_by_id(user_id)
        if user is not None:
            self.session.delete(user)
            self.session.commit()
        else:
            raise ValueError(f"User with id {user_id} does not exist.")