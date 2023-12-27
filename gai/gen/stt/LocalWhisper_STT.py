import torch,gc,os
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.common.utils import get_config_path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import PosixPath

class LocalWhisper_STT:

    def __init__(self,model_config):
        self.config = model_config
        self.client = None
        pass

    def load(self):
        logger.info(f"Loading model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = os.path.join(get_config_path(),self.config['model_path'])
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        self.client = AutoProcessor.from_pretrained(model_id)
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

        chunk_length_s=30
        if chunk_length_s in model_params:
            chunk_length_s=model_params["chunk_length_s"]

        batch_size=16
        if batch_size in model_params:
            batch_size=model_params["batch_size"]

        max_new_tokens=128
        if max_new_tokens in model_params:
            max_new_tokens=model_params["max_new_tokens"]

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.client.tokenizer,
            feature_extractor=self.client.feature_extractor,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        return self.pipe(file, generate_kwargs={"language": "english"})