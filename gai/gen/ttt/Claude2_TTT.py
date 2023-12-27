import os,json
from dotenv import load_dotenv
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from typing import List
from uuid import uuid4
from datetime import datetime
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta

class Claude2_TTT:

    param_whitelist=[
        "max_tokens_to_sample",
        "temperature",
        "stream"
    ]

    def get_model_params(self, **kwargs):
        params={
            "max_tokens_to_sample":25
        }        
        return {**params,**kwargs}

    def __init__(self,model_config):
        self.client = None
        self.config = model_config

    def load(self):
        from dotenv import load_dotenv        
        load_dotenv()
        class MissingAnthropicApiKeyException(Exception):
            pass        
        if 'ANTHROPIC_API_KEY' not in os.environ:
            msg = "ANTHROPIC_API_KEY not found in environment variables. Claude will not be available."
            logger.warning(msg)
            raise MissingAnthropicApiKeyException(msg)
        self.client=Anthropic()

    def unload(self):
        return self

    def _remove_template(self,output):
        return output

    def _apply_template(self,prompt: List):
        content = generators_utils.chat_list_to_string(prompt)
        prompt_template = "\n\nHuman: {content}\n\nAssistant:"
        prompt = prompt_template.format(content=content)
        return prompt
    
    def _generating(self,prompt,**model_params):
        id = str(uuid4())
        response = self.client.completions.create(
            model="claude-2",
            prompt=prompt,
            stream=False,
            **model_params
            )
        response = self.parse_generating_output(id=id, output=response.completion, finish_reason=response.stop_reason)
        return response
    
    def parse_generating_output(self, id, output,finish_reason):

        # Map the finish_reason to the same values as the OpenAI API
        if finish_reason == "stop_sequence":
            finish_reason = "stop"
        if finish_reason == "max_tokens":
            finish_reason = "length"

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
                    finish_reason= finish_reason,
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

    def _streaming(self,prompt,**model_params):
        id = str(uuid4())
        for output in self.client.completions.create(
                model="claude-2",
                prompt=self.prompt,
                stream=True,
                **model_params
            ):
            yield self.parse_chunk_output(
                id=id, 
                output=output.completion 
                )
        yield self.parse_chunk_output(
            id=id, 
            output=output.completion, 
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
            logger.error(f"Claude2Engine: error={e} id={id} output={output} finish_reason={finish_reason}")
            raise Exception(e)



    def create(self,messages,**model_params):
        self.prompt = self._apply_template(messages)
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
    
    def token_count(self,text):
        output= self.client.count_tokens(text)
        return output
