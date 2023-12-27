from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

class ITT:
    
    def __init__(self,generator_name):
        self.generator_name = generator_name
        self.config = generators_utils.load_generators_config()[generator_name]
        if self.config['engine'] == 'Llava_ITT':
            from gai.gen.itt.Llava_ITT import Llava_ITT
            self.itt = Llava_ITT(self.config)
        elif self.config['engine'] == 'Vision_ITT':
            from gai.gen.itt.Vision_ITT import Vision_ITT
            self.itt = Vision_ITT(self.config)
        else:
            logger.error("Image to Speech engine not supported")
            raise Exception("Image to Speech engine not supported")

    def create(self,**model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        return self.itt.create(**model_params)

    def gen(self, **model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        return self.itt.gen(**model_params)

    def load(self):
        if "engine" in self.config:
            logger.info(f"Using itt model {self.config['engine']}...")
        self.itt.load()
        return self

    def unload(self):
        logger.info(f"Unloading itt model...")
        self.itt.unload()
