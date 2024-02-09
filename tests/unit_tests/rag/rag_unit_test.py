import unittest
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from gai.gen.rag.models.IndexedDocument import IndexedDocument
from gai.gen.rag.models.Base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from gai.gen.rag.Repository import Repository
import logging

from gai.common.utils import get_config, get_config_path

class RagUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # db_dir = get_config_path()
        # db_file = get_config()["gen"]["rag"]["sqlite"]["path"]
        # db_path=os.path.join(db_dir, db_file)
        # cls.repo = Repository(f'sqlite:///{db_path}')
        cls.repo = Repository()

    def test_create_indexed_document(self):

        # Preparation
        doc = IndexedDocument()
        doc.CollectionName = 'demo'
        doc.ChunkSize = 1000
        doc.Overlap = 100
        doc.Title = 'Test Document'
        doc.FileName = 'attention-is-all-you-need.pdf'
        doc.Source = 'https://arxiv.org/abs/1706.03762'
        doc.Abstract = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data"""
        doc.Authors = 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin'
        doc.Publisher = 'arXiv'
        doc.PublishedDate = datetime.strptime('2017-June-12','%Y-%B-%d').date()
        doc.Comments = 'This is a test document'
        doc.IsActive = True
        doc.CreatedAt = datetime.now()
        doc.UpdatedAt = datetime.now()

        # Act
        chunks = ['0b7c01468ca7505ef751e06647b17d0475d51813c3a1925590df81a7a4b5a3a7', '0c494f2a001a0666487734f76141fd75fb0d570c4fcc9e1463cc31c8bf2eeaca', '0deef6826b50e155ce7ccec0048190136b60fd707df2b207e7097851175f47de', '1678a6ce913a2c99d2a6f103a010cbc90a3c5ac8e6c78bf636cfbda9507af0cf', '17ec7d5812b3efd303a797663196ab70ed36e913908f232aca68c234f13332d2', '1eaa155a661d38285024eb5b5c217bc930404bf943d8ebb445b87dedd1ddbafd', '222bcda88c65b1f572c0339f697a693e75ba801d14f7a4bb588a758163fae40e', '243a269e5895fef77afcc4dbbbe74aeef919944018b2ae2e2bd6537ed6640b42', '2487f6e4cc193ca3b9c7f0b961b0d47f5d28dd0ca7df5cfb6e8f276f452ed834', '2c1abad7e99a713f0f8ac27f0178ec5fadcfbf894dd2096383dfc49bf1bf8059', '32793c76bc2d62b3a59ce91757fbe0fac942358be40fbcd3848cc87da722650e', '3606915a8c2b0764daa665f3f77b3c8e789e96da38ab037ae21f786ecd6d0fae', '4eacb76b09d35507a913d078faa49f507beda51f65c77f40c81bc56f9b5ba648', '5255ad964d5937cfbd77330e4ade51be36591644e1184594bcb73ed1c77af8ca', '53bfb1d80c93baf656446d5c5257e79163a771f414351902244e6d2f2e6c88c9', '55e2cfde42f2e10abd85d1c5a0135cdf89138472bf18e390d768cc5eca919389', '57d3f5af7dd4bf93488fe5e837e22591b53513ab7faa9199ef91a066bae13f31', '5d61c4ead1186f6de6909eca57f6675c789e43b09cfec6b738cb802b7b4eea69', '5f400f3e27f615c6663d4f355745b58ecc9730380d74726e13e8731251c67e16', '6ade840086b93ffd3fec0f1176c6eb08771b2e39823f5eabfd2800d4c862c09e', '6c48a2efa90751540386bfa7af8d74abd5bad882473954d9c73ecb68088db872', '6f86ccc47643496442273b3c4d397661c5063972750efee125082385f113099a', '717a762f9b14c2ab814b6ecc286d71ff0fb0c469476bd0e6e3cd75af70745f9b', '73ffba02a444eb4ded0bb5f1e6258660ede0897b8aff611bec1b5e2c269c9b68', '7991669cd66653413bbd013f935260fe5bae11d6fbd8400e5995d2ffd58fc499', '867b511eb966cd9e89e29416be744feb6cc7df43209a07f6cf160f23c218f065', '87633d7b1d93651484d23cecb762900b9c97db67fdc998f60375feac4e5df67a', '8aa8cc17c40a9a090458c6725cd16cdae0cdf5d311e7b935affb461f1877a287', '91768e56ebeb2e2e1ed60b945c6a117b476eb3ad60e42176e4b6db7a842246f7', '9396036a0b1107937619fbb66b16379754b446ec04adfda74e95e22b0ba8ca2b', '94f0f70f5a0ec555696a1bac479d55533734d89fcfa4913637ecfa56b5d3f5bd', 'a3cbfc46792960e682376932bc02b7a43508c7f5dcf97934dd2be978c8091889', 'b14b2128be165ff7195c08914cc11f9203888392776b80867ff7594e4b49df1f', 'b84387fd63d314d528e1a1bd85e3e4edbeb2f99c3ef241e8c400bb5145cf418d', 'b85f84053b95c63d3954100e81de5d493b3e6702415fe294c87eb1ae76204a8e', 'b9210565cbd269027dd0b19e91c3ab7db415670c088c0cada970c970a165f784', 'c2bde86c22b51579f1897a6888dc70fc903ce8a9f73c64f0d99975ecf56a4b49', 'c2e5d965858b02bb42ca2b31b3f6372aa7f1da2966fcc137ded5643b71bf3696', 'c9d4145569a23677b5d47a9af98d7aa3eadc9d333b295d4bd97a04efb1cd5134', 'cead4552077c85ef19d06a425c2297547c513718a8fabac782ef824f0332da71', 'd040a9e16a818fd6598483721a043a3a22bf9dca24bf35d0cf4e8ae4cf728e81', 'd73b34db9990408b438dfaf4348593d93c87b3922f23051f458459e2d0d33072', 'd96cc56503db04cfa4f6c10fd39aa3250f6e1a3c0ff9b85a119585b8a7d92df4', 'dcfa12bb04a5a0b40a5bfee4bab4db230c9b64c0783ce1759c924b15364e5b0b', 'df8c6b5c94e406bcfa45293d0e4e82866a12f9b07a2b5cf11cd835d155eb5a93', 'e1d2af6e9e1234d78a9424edf83fed3c72d766851bbdc42771de48d6f5caec97', 'e2e7c6d16908840896d5a3da14624d6c98b7b76f5a3eb1cccb68ee6e9f16b972', 'e3e4ef022d7570549eaa666db278bd01cac1453b4838ae5ea07137371adf8277', 'e91e7677b426de262824a4173528532dfd0f08a8ff550746a28ba414c03fdc6c', 'eafa20be8f5a92207f25cfd85774fbb436f53d5851c16d8bfac8cab47e0b2559', 'eb22ff9a2ca2b8999c1737e6ba6a186585faa3be8653801adfd0accc5def4ad0', 'f5b54668d2357185abda3d81ceda8d1218b230100a483e58e45c94d8765432ef']
        doc_id = self.repo.create_document(doc,chunks)

        # Assert
        engine = self.repo.engine
        Session = sessionmaker(bind=engine)
        session = Session()
        retrieved_doc = session.query(IndexedDocument).filter_by(Id=doc_id).first()
        self.assertEqual(retrieved_doc is not None, True)

    def test_list_documents(self):
        # Act
        docs = self.repo.list_documents()

        # Assert
        self.assertEqual(len(docs) > 0, True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(exit=False)
