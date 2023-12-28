import torch, gc, re,os
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.common.utils import get_config_path
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator as ExLlamaGen
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from uuid import uuid4
from datetime import datetime
from typing import List

class ExLlama_TTT:

    param_whitelist=[
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical",
        "token_repetition_penalty_max",
        "token_repetition_penalty_sustain",
        "token_repetition_penalty_decay",
        "beams",
        "beam_length",
        "max_new_tokens",
        "stream"
        ]

    def get_model_params(self, **kwargs):
        params={
            "max_new_tokens":25,
            "temperature":0.7,
            "top_p": 1,
            "top_k": 3
        }        
        return {**params,**kwargs}

    def __init__(self,model_config):
        if (model_config is None):
            raise Exception("exllama_engine: model_config is required")
        if "model_path" not in model_config or model_config["model_path"] is None:
            raise Exception("exllama_engine: model_path is required")
        if "model_basename" not in model_config or model_config["model_basename"] is None:
            raise Exception("exllama_engine: model_basename is required")

        self.config = model_config
        self.model_filepath = os.path.join(get_config_path(), model_config["model_path"], model_config["model_basename"])+".safetensors"
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None

    def load(self):
        logger.info(f"exllama_engine.load: Loading model from {self.model_filepath}")

        # model
        model_config_path = os.path.join(get_config_path(),self.config["model_path"], 'config.json')

        exllama_config = ExLlamaConfig(model_config_path)        
        exllama_config.max_seq_len = self.config["max_seq_len"]
        exllama_config.model_path = self.model_filepath
        self.model = ExLlama(exllama_config)

        # tokenizer
        tokenizer_path= os.path.join(get_config_path(),self.config["model_path"], 'tokenizer.model')
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)         

        # generator
        self.client = ExLlamaGen(self.model, self.tokenizer, ExLlamaCache(self.model))

        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.client
            del self.prompt
        except :
            pass
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self, text):
        if self.tokenizer is None:
            raise Exception("exllama_engine: tokenizer is not loaded")
        encoded=self.tokenizer.encode(text)
        return len(encoded.tolist()[0])

    def get_response(self,output,ai_role="ASSISTANT"):
        return re.split(rf'{ai_role}:', output, flags=re.IGNORECASE)[-1].strip().replace('\n\n', '\n').replace('</s>', '')
    
    def _init_settings(self,model_params):
        self.client.settings.temperature = model_params["temperature"] if "temperature" in model_params and model_params["temperature"] is not None else self.client.settings.temperature
        self.client.settings.top_p = model_params["top_p"] if "top_p" in model_params and model_params["top_p"] is not None else self.client.settings.top_p
        self.client.settings.min_p = model_params["min_p"] if "min_p" in model_params and model_params["min_p"] is not None else self.client.settings.min_p
        self.client.settings.top_k = model_params["top_k"] if "top_k" in model_params and model_params["top_k"] is not None else self.client.settings.top_k

        self.client.settings.token_repetition_penalty_max = model_params["token_repetition_penalty_max"] if "token_repetition_penalty_max" in model_params and model_params["token_repetition_penalty_max"] is not None else self.client.settings.token_repetition_penalty_max
        self.client.settings.token_repetition_penalty_sustain = model_params["token_repetition_penalty_sustain"] if "token_repetition_penalty_sustain" in model_params and model_params["token_repetition_penalty_sustain"] is not None else self.client.settings.token_repetition_penalty_sustain
        self.client.settings.token_repetition_penalty_decay = model_params["token_repetition_penalty_decay"] if "token_repetition_penalty_decay" in model_params and model_params["token_repetition_penalty_decay"] is not None else self.client.settings.token_repetition_penalty_decay
        
        self.client.settings.typical = model_params["typical"] if "typical" in model_params and model_params["typical"] is not None else self.client.settings.typical
        self.client.settings.beams = model_params["beams"] if "beams" in model_params and model_params["beams"] is not None else self.client.settings.beams
        self.client.settings.beam_length = model_params["beam_length"] if "beam_length" in model_params and model_params["beam_length"] is not None else self.client.settings.beam_length

    def _generate_simple(self, prompt, max_new_tokens = 128):

        self.client.end_beam_search()

        ids, mask = self.client.tokenizer.encode(prompt, return_mask = True, max_seq_len = self.model.config.max_seq_len)
        self.client.gen_begin(ids, mask = mask)

        max_new_tokens = min(max_new_tokens, self.client.model.config.max_seq_len - ids.shape[1])

        finish_reason="length"
        eos = torch.zeros((ids.shape[0],), dtype = torch.bool)
        for i in range(max_new_tokens):
            token = self.client.gen_single_token(mask = mask)
            for j in range(token.shape[0]):
                if token[j, 0].item() == self.client.tokenizer.eos_token_id: eos[j] = True
            if eos.all(): 
                finish_reason="stop"
                break

        text = self.client.tokenizer.decode(self.client.sequence[0] if self.client.sequence.shape[0] == 1 else self.client.sequence)
        return {"output":text, "finish_reason":finish_reason}

    def _generating(self, prompt,**model_params):
        logger.debug(f"exllama_engine.generate: prompt={prompt}")

        # Map "max_tokens" to "max_new_tokens" to be compatible with OpenAI's API. We do not want to filter this off.
        if "max_tokens" in model_params and model_params["max_tokens"] is not None:
            model_params["max_new_tokens"]=model_params.pop("max_tokens")

        # Temperature approach 0 but cannot be 0
        if "temperature" in model_params and model_params["temperature"]==0:
            model_params["temperature"]=10e-10

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        logger.debug(f"exllama_engine.generate: model_params={model_params}")
        
        input_count=self.token_count(prompt)
        logger.debug(f"exllama_engine.generate: input token count={input_count}")

        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200
        response = self._generate_simple(prompt,max_new_tokens=max_new_tokens)
        
        logger.debug(f"exllama_engine.generate: raw output={response}")
        
        # Prepare response
        id = str(uuid4())
        response = self.parse_generating_output(id=id, output=response['output'], finish_reason=response['finish_reason'])
        return response

    # SAMPLE RESPONSE:
    # ChatCompletion(
    #    id='chatcmpl-8YquW981VnABKGP0HhIigugRttQWu', 
    #    choices=[
    #        Choice(
    #           finish_reason='length', 
    #           index=0, 
    #           logprobs=None, 
    #           message=ChatCompletionMessage(
    #               content='Once upon a time in a bustling city lived a scruffy, little stray dog named Baxter. Despite his hardships, Baxter had a heart full of hope and would wag his tail at every passerby, hoping someone would take him home. One icy winter day, a kind-hearted woman named Lucy noticed him shivering in a corner. Lucy, who had recently lost her beloved pet, felt an immediate connection with Baxter. Overwhelmed with compassion, she decided to adopt him right then. From that day', 
    #               role='assistant', 
    #               function_call=None, 
    #               tool_calls=None))], 
    #   created=1703317232, 
    #   model='gpt-4-0613', 
    #   object='chat.completion', 
    #   system_fingerprint=None, 
    #   usage=CompletionUsage(completion_tokens=100, prompt_tokens=34, total_tokens=134))
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

    def _streaming(self,prompt,**model_params):

        new_text = ""
        last_text = ""
        _full_answer = ""

        logger.debug(f"exllama_engine.streaming: prompt={prompt}")
        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        logger.debug(f"model_params: {model_params}")

        input_count=self.token_count(prompt)
        logger.debug(f"exllama_engine.streaming: input token count={input_count}")

        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200

        self.client.end_beam_search()
        ids = self.tokenizer.encode(prompt)
        self.client.gen_begin_reuse(ids)
        id = str(uuid4())
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            # Get new token by taking difference from last response:
            new_token = new_text.replace(last_text, "")
            last_text = new_text

            # [End conditions]:
            if token.item() == self.tokenizer.eos_token_id:
                logger.debug(f"exllama_engine.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                yield self.parse_chunk_output(id,new_token, "stop")
                return

            #if break_on_newline and 
            # could add `break_on_newline` as a GenerateRequest option?
            #if token.item() == tokenizer.newline_token_id:
            #    logger.debug(f"newline_token_id: {tokenizer.newline_token_id}")
            #    yield self.parse_last_chunk(new_token, "stop")
            #    return

            if i == max_new_tokens - 1:
                logger.debug(f"exllama_engine.streaming: stopped by max_new_tokens: {max_new_tokens}")
                yield self.parse_chunk_output(id,new_token, "length")
                self.client.end_beam_search() 
                return

            yield self.parse_chunk_output(id,new_token)

        # all done:
        self.client.end_beam_search() 
        return

    # Sample:
    #ChatCompletionChunk(
    #     id='chatcmpl-8YqIOmXu1WLlYcYeMbhPg6yYWBQ1u', 
    #     choices=[
    #          Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), 
    #          finish_reason=None, 
    #          index=0, 
    #          logprobs=None)],
    #     created=1703314868,
    #     model='gpt-4-0613',
    #     object='chat.completion.chunk',
    #     system_fingerprint=None)
    #.....
    #ChatCompletionChunk(
    #     id='chatcmpl-8YqIOmXu1WLlYcYeMbhPg6yYWBQ1u',
    #     choices=[
    #          Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), 
    #          finish_reason='length', 
    #          index=0,
    #          logprobs=None)], 
    #     created=1703314868, 
    #     model='gpt-4-0613', 
    #     object='chat.completion.chunk', 
    #     system_fingerprint=None)
    def parse_chunk_output(self, id, output,finish_reason=None):
        created = int(datetime.now().timestamp())
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

    def _apply_template(self, prompt:List):
        content = generators_utils.chat_list_to_string(prompt)
        prompt_template = "<s>[INST] {prompt} [/INST]"
        prompt = prompt_template.format(prompt=content)
        return prompt

    def _remove_template(self, output:str):
        prompt_template = r'<s>\[INST\].*?\[/INST\]\s*'
        return re.sub(prompt_template, '', output, flags=re.S)

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

        