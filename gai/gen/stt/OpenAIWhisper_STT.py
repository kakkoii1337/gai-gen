import os, openai
from openai import OpenAI
from gai.common import generators_utils, logging
logger = logging.getLogger(__name__)

class OpenAIWhisper_STT:

    def __init__(self,gai_config):
        self.gai_config = gai_config
        self.client = None
        pass

    def load(self):
        from dotenv import load_dotenv        
        load_dotenv()
        class MissingOpenAIApiKeyException(Exception):
            pass
        if 'OPENAI_API_KEY' not in os.environ:
            msg = "OPENAI_API_KEY not found in environment variables. GPT-4 will not be available."
            logger.warning(msg)
            raise MissingOpenAIApiKeyException(msg)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI()
        return self

    def unload(self):
        self.client = None
        return self

    def create(self,**model_params):
        if not self.client:
            self.load()

        if "file" not in model_params:
            raise Exception("Missing file parameter")
        
        file = model_params["file"]

        # If file is a bytes object, we need to write it to a temporary file then pass the file object to the API
        if isinstance(file,bytes):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp:
                temp.write(file)
                temp.flush()
                temp.seek(0)
                model_params["file"] = temp.file
                response= self.client.audio.transcriptions.create(model='whisper-1',**model_params)
        else:
            response = self.client.audio.transcriptions.create(model='whisper-1',**model_params)
        return response
