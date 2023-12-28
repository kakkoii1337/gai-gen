import threading
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

class Gaigen:
    __instance = None       # singleton

    @staticmethod 
    def GetInstance(generator_name=None):
        """Static method to access this singleton class's instance."""
        if Gaigen.__instance == None:
            Gaigen()
        return Gaigen.__instance

    def __init__(self):
        """Virtually private constructor."""
        if Gaigen.__instance is not None:
            raise Exception("Gaigen: This class is a singleton!")
        else:
            self.config = generators_utils.load_generators_config()
            self.generator_name = None
            self.generator = None
            self.semaphore = threading.Semaphore(1)     # for thread safety, using Semaphore allows for easier upgrade to support multiple generators in the future
            Gaigen.__instance = self

    # This is idempotent
    def load(self, generator_name):

        if generator_name is None:
            logger.error("Gaigen.load: generator_name parameter is required.")
            raise Exception("Gaigen.load: generator_name parameter is required.")
        
        if self.generator_name  == generator_name:
            logger.debug("Gaigen.load: Generator is already loaded. Skip loading.")
            return self.generator

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
        else:
            logger.error(f"Gaigen.load: The generator_type {generator_type} is not supported.")
            raise Exception(f"Gaigen.load: The generator_type {generator_type} is not supported.")
        
        try:
            logger.info(f"Gaigen: Loading generator {generator_name}...")
            self.generator.load()
            self.generator_name = generator_name
            return self.generator
        except Exception as e:
            logger.error(f"Gaigen: Error loading generator {generator_name}: {e}")
            raise e
       

    def unload(self):
        if self.generator is not None:
            self.generator.unload()
            self.generator = None

    def create(self,**model_params):
        with self.semaphore:
            self.load()
            return self.generator.create(**model_params)
    
    def token_count(self,text):
        self.load()
        if hasattr(self.generator, 'token_count'):
            return self.generator.token_count(text)
        raise Exception("token_count is not supported by this generator.")

    def get_token_ids(self,text):
        self.load()
        if hasattr(self.generator, 'get_token_ids'):
            return self.generator.get_token_ids(text)
        raise Exception("get_token_ids is not supported by this generator.")
