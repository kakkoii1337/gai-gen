import threading
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

class Gaigen:
    __instance = None       # singleton

    @staticmethod 
    def GetInstance():
        """Static method to access this singleton class's instance."""
        if Gaigen.__instance == None:
            Gaigen()
        return Gaigen.__instance

    def __init__(self):
        """Virtually private constructor."""
        if Gaigen.__instance is not None:
            raise Exception("Gaigen: This class is a singleton! Access using GetInstance().")
        else:
            self.config = generators_utils.load_generators_config()
            self.generator_name = None
            self.generator = None
            self.semaphore = threading.Semaphore(1)     # for thread safety, using Semaphore allows for easier upgrade to support multiple generators in the future
            Gaigen.__instance = self

    # This is idempotent
    def load(self,generator_name):

        if generator_name is None:
            logger.error("Gaigen.load: generator_name parameter is required.")
            raise Exception("Gaigen.load: generator_name parameter is required.")
        
        if self.generator_name  == generator_name:
            logger.debug("Gaigen.load: Generator is already loaded. Skip loading.")

        if self.generator_name and self.generator_name != generator_name:
            logger.debug("Gaigen.load: New generator_name specified, unload current generator.")
            self.generator.unload()

        generator_type = self.config[generator_name]["type"]
        if generator_type == "ttt":
            from gai.gen.ttt import TTT
            self.generator = TTT(generator_name=generator_name)
        elif generator_type == "tts":
            from gai.gen.tts import TTS
            self.generator = TTS(generator_name=generator_name)
        elif generator_type == "stt":
            from gai.gen.stt import STT
            self.generator = STT(generator_name=generator_name)
        elif generator_type == "itt":
            from gai.gen.itt import ITT
            self.generator = ITT(generator_name=generator_name)
        elif generator_type == "rag":
            from gai.gen.rag import RAG
            self.generator = RAG()
        else:
            logger.error(f"Gaigen.load: The generator_type {generator_type} is not supported.")
            raise Exception(f"Gaigen.load: The generator_type {generator_type} is not supported.")
        try:
            logger.info(f"Gaigen: Loading generator {generator_name}...")
            self.generator.load()
            self.generator_name = generator_name
            return self
        except Exception as e:
            logger.error(f"Gaigen: Error loading generator {generator_name}: {e}")
            raise e

    def unload(self):
        if self.generator is not None:
            self.generator.unload()
            self.generator = None
        return self

    def create(self,**model_params):
        if self.generator is None:
            logger.error("Gaigen.create: Generator is not loaded.")
            raise Exception("Gaigen.create: Generator is not loaded.")
        with self.semaphore:
            return self.generator.create(**model_params)
    
    def token_count(self,text):
        if self.generator is None:
            logger.error("Gaigen.create: Generator is not loaded.")
            raise Exception("Gaigen.create: Generator is not loaded.")
        if hasattr(self.generator, 'token_count'):
            return self.generator.token_count(text)
        raise Exception("token_count is not supported by this generator.")

    def get_token_ids(self,text):
        if self.generator is None:
            logger.error("Gaigen.create: Generator is not loaded.")
            raise Exception("Gaigen.create: Generator is not loaded.")
        if hasattr(self.generator, 'get_token_ids'):
            return self.generator.get_token_ids(text)
        raise Exception("get_token_ids is not supported by this generator.")

    def index(self, collection_name, text, path_or_url, metadata={"source":"unknown"}, chunk_size=None, chunk_overlap=None):
        if self.generator is None:
            logger.error("Gaigen.create: Generator is not loaded.")
            raise Exception("Gaigen.create: Generator is not loaded.")

        if self.generator.generator_name != "rag":
            logger.error(f"Gaigen.index: The generator {self.generator.generator_name} does not support indexing.")
            raise Exception(f"Gaigen.index: The generator {self.generator.generator_name} does not support indexing.")
        with self.semaphore:
            return self.generator.index(collection_name, text, path_or_url, metadata, chunk_size, chunk_overlap)
        
    def retrieve(self, collection_name, query_texts, n_results=None):
        if self.generator is None:
            logger.error("Gaigen.create: Generator is not loaded.")
            raise Exception("Gaigen.create: Generator is not loaded.")

        if self.generator.generator_name != "rag":
            logger.error(f"Gaigen.retrieve: The generator {self.generator.generator_name} does not support retrieval.")
            raise Exception(f"Gaigen.retrieve: The generator {self.generator.generator_name} does not support retrieval.")
        with self.semaphore:
            return self.generator.retrieve(collection_name, query_texts, n_results)