import os, openai
from openai import OpenAI
from gai.common import generators_utils, logging
logger = logging.getLogger(__name__)
from typing import List

class OpenAI_TTT:

    param_whitelist=[
        'frequency_penalty',
        'presence_penalty',
        'temperature',
        'max_tokens',
        'logit_bias',
        'stream'
        'top_p',
        'stop',
        'n',
        ]

    def get_model_params(self, **kwargs):
        params = {
            'max_tokens': 25,
            'temperature': 0.7,
            'top_p': 1,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'stop': None,
            'logit_bias': {},
            'n': 1
        }
        params = {**params,**kwargs}
        return params

    def __init__(self,model_config):
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
        return self

    def apply_template(self,prompt: List):
        content = generators_utils.chat_list_to_string(prompt)
        prompt = [
            {"role":"user","content":content},
            {"role":"assistant","content":""}
            ]
        return prompt

    def create(self,messages,**model_params):
        if not self.client:
            self.load()

        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        stream = model_params.pop('stream',False)

        self.prompt = self.apply_template(messages)

        if not stream:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.prompt,
                stream=stream,
                **model_params
            )
            return response

        return (chunk for chunk in self.client.chat.completions.create(
            model="gpt-4",
            messages=self.prompt,
            stream=stream,
            **model_params
        ))
