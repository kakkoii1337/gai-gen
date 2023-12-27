from gai.gen.stt.OpenAIWhisper_STT import OpenAIWhisper_STT
from gai.gen.stt.LocalWhisper_STT import LocalWhisper_STT
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

class STT:
    def __init__(self,generator_name):
        self.generator_name = generator_name
        self.config = generators_utils.load_generators_config()[generator_name]
        if self.config['engine'] == 'OpenAIWhisper_STT':
            self.transcriptions = OpenAIWhisper_STT(self.config)
        elif self.config['engine'] == 'LocalWhisper_STT':
            self.transcriptions = LocalWhisper_STT(self.config)
        else:
            logger.error("Speech to Text engine not supported")
            raise Exception("Speech to Text engine not supported")
    
    def load(self):
        if "engine" in self.config:
            logger.info(f"Using stt model {self.config['engine']}...")
        self.transcriptions.load()
        return self

    def unload(self):
        logger.info(f"Unloading stt model...")
        self.transcriptions.unload()


    def create(self,**model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        
        return self.transcriptions.create(**model_params)
