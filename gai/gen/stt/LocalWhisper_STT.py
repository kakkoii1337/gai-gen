import torch,gc,os
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.common.utils import get_config_path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import PosixPath

class LocalWhisper_STT:

    def __init__(self,gai_config):
        self.gai_config = gai_config
        self.client = None
        pass

    def load(self):
        logger.info(f"Loading model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = os.path.join(get_config_path(),self.gai_config['model_path'])
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        self.client = AutoProcessor.from_pretrained(self.model_path)
        return self

    def unload(self):
        try:
            del self.model
            del self.pipe
            del self.client
        except :
            pass
        self.model = None
        self.pipe = None
        self.client = None
        gc.collect()
        torch.cuda.empty_cache()

    def create(self,**model_params):
        if not self.client:
            self.load()

        file = model_params.pop("file",None)
        if file is None:
            raise Exception("Missing audio data")
        import io
        if (isinstance(file, io.IOBase)):
            file = file.read()
        elif (isinstance(file, PosixPath)):
            file = file.read_bytes()

        model_params = {**self.gai_config["hyperparameters"],**model_params}

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.client.tokenizer,
            feature_extractor=self.client.feature_extractor,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
            **model_params
        )
        
        return self.pipe(file, generate_kwargs={"language": "english"})