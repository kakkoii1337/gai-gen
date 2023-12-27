from llama_cpp import Llama
from llama_cpp._utils import suppress_stdout_stderr
from gai.common import generators_utils, logging
from gai.common.utils import get_config_path
import os,sys,torch,gc,re
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from uuid import uuid4
from datetime import datetime
from typing import List
logger = logging.getLogger(__name__)

class LlamaCpp_TTT:

    param_whitelist=[
        'max_new_tokens',
        'stopping_criteria',
        'temperature',
        'top_k',
        'top_p',
        'stream'
        ]

    def get_model_params(self, **kwargs):
        # llama_cpp uses max_tokens instead of max_new_tokens
        # Transform max_new_tokens to max_tokens if max_new_tokens is in kwargs
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"]=kwargs["max_new_tokens"]
            kwargs.pop("max_new_tokens")

        params = {
            'max_tokens': 25,
            'temperature': 1.31,
            'top_k': 49,
            'top_p': 0.14,
        }
        params = {**params,**kwargs}
        return params

    def __init__(self,model_config):
        if (model_config is None):
            raise Exception("llamacpp_engine: model_config is required")
        if "model_path" not in model_config or model_config["model_path"] is None:
            raise Exception("llamacpp_engine: model_path is required")
        if "model_basename" not in model_config or model_config["model_basename"] is None:
            raise Exception("llamacpp_engine: model_basename is required")

        self.config = model_config
        self.model_filepath = os.path.join(get_config_path(), model_config["model_path"], model_config["model_basename"])
        self.model = None
        self.tokenizer = None
        self.client = None

    def load(self):
        logger.info(f"exllama_engine.load: Loading model from {self.model_filepath}")
        with suppress_stdout_stderr():
            self.client = Llama(model_path=self.model_filepath, verbose=False, n_ctx=self.config["max_seq_len"])
        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.client
        except :
            pass
        self.model = None
        self.tokenizer = None
        self.client = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self,text):
        return len(self.client.tokenize(text.encode()))

    def _generating(self,prompt, **model_params):
        response = None
        with suppress_stdout_stderr():
            response = self.client(prompt,**model_params)

        # Prepare response
        id = str(uuid4())
        response = self.parse_generating_output(id=id, output=response['choices'][0]['text'], finish_reason=response['choices'][0]['finish_reason'])
        return response

    def _apply_template(self, prompt:List):
        prompt = generators_utils.chat_list_to_string(prompt)
        return prompt

    def _remove_template(self, output:str):
        return output

    def parse_generating_output(self, id, output,finish_reason):
        output = self._remove_template(output)
        prompt_tokens = self.token_count(self.prompt)
        completion_tokens = self.token_count(output)
        total_tokens = prompt_tokens + completion_tokens
        created = int(datetime.now().timestamp())
        response = ChatCompletion(
            id=id,                
            choices=[
                Choice(
                    # "stop","length","content_filter"
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=output, 
                        role='assistant', 
                        function_call=None, 
                        tool_calls=None
                    ))
            ],
            created=created,
            model=self.config["model_name"],
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,total_tokens=total_tokens)
            )
        return response

    def _streaming(self,prompt,ai_role="ASSISTANT",**model_params):
        id = str(uuid4())
        with suppress_stdout_stderr():
            for chunk in self.client(prompt,stream=True,**model_params):
                yield self.parse_chunk_output(
                    id=id,
                    output=chunk['choices'][0]['text']
                )
        yield self.parse_chunk_output(
            id=id, 
            output=chunk['choices'][0]['text'], 
            finish_reason="stop"
            )                

    def parse_chunk_output(self, id, output,finish_reason=None):
        created = int(datetime.now().timestamp())
        try:
            response = ChatCompletionChunk(
                id=id,                
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=output, function_call=None, role='assistant', tool_calls=None),
                        # "stop","length","content_filter"
                        finish_reason=finish_reason,
                        index=0,
                        logprobs=None,
                        message=output
                        )
                ],
                created=created,
                model=self.config["model_name"],
                object="chat.completion.chunk",
                system_fingerprint=None,
                usage=None
                )
            return response
        except Exception as e:
            logger.error(f"LlamaCppEngine: error={e} id={id} output={output} finish_reason={finish_reason}")
            raise Exception(e)

    def create(self,messages,**model_params):
        self.prompt=self._apply_template(messages)
        if not self.client:
            self.load()

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        stream = model_params.pop("stream", False)

        if not stream:
            response = self._generating(
                prompt=self.prompt,
                **model_params
            )
            return response

        return (chunk for chunk in self._streaming(
            prompt=self.prompt,
            **model_params
        ))    