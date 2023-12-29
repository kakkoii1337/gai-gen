import torch,os,gc
from gai.common import generators_utils, logging
from gai.common.utils import get_config_path
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,StoppingCriteriaList, TextStreamer, TextIteratorStreamer
from threading import Thread
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from uuid import uuid4
from datetime import datetime
from typing import List
import re

class Transformers_TTT:

    param_whitelist=[
        'max_new_tokens',
        'temperature',
        'top_k',
        'top_p',
        'do_sample',
        'stream'
        ]

    def get_model_params(self, **kwargs):
        params = {
            'max_new_tokens': 25,
            'temperature': 1.31,
            'top_k': 49,
            'top_p': 0.14,
            'do_sample': True
        }
        params = {**params,**kwargs}
        return params

    def __init__(self,gai_config):
        if (gai_config is None):
            raise Exception("transformers_engine: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("transformers_engine: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("transformers_engine: model_basename is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_config_path(), gai_config["model_path"], gai_config["model_basename"])
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load(self):
        logger.info(f"transformers_enginer: Loading model from {self.gai_config['model_path']}")

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(get_config_path(),self.gai_config['model_path']))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        max_memory = f'{40960}MB'
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(get_config_path(),self.gai_config['model_path']), 
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: max_memory for i in range(n_gpus )},)
        return self

    def unload(self):
        logger.info(f"transformers_enginer: Unloading model...")        
        try:
            del self.model
            del self.tokenizer
            del self.generator
        except :
            pass
        self.model = None
        self.tokenizer = None
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self,text):
        return len(self.tokenizer.tokenize(text))
    
    def get_token_ids(self,text:str):
        return self.tokenizer.encode(text)

    def _apply_template(self, prompt:List):
        prompt = generators_utils.chat_list_to_string(prompt)
        return prompt

    def _remove_template(self, output:str):
        match = list(re.finditer(r'\n.+:\s', output))
        if match:
            last_match = match[-1]
            return output[last_match.end():]
        else:
            return output

    def _generating(self,prompt, **model_params):
        logger.debug(f"transformers_engine.generate: prompt={prompt}")

        input_ids = self.tokenizer(prompt,return_tensors="pt",add_special_tokens=True).input_ids.cuda()
        generated = self.model.generate(input_ids,**model_params)
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Prepare response
        id = str(uuid4())
        response = self.parse_generating_output(id=id, output=response, finish_reason='stop')
        return response

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
            model=self.gai_config["model_name"],
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,total_tokens=total_tokens)
            )
        return response

    def _streaming(self,prompt, **model_params):
        logger.debug(f"transformers_engine.generate: prompt={prompt}")

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        logger.debug(f"transformers_engine.generate: model_params={model_params}")

        input_count=self.token_count(prompt)
        logger.debug(f"transformers_engine.generate: input token count={input_count}")

        input_ids = self.tokenizer(prompt,return_tensors="pt",add_special_tokens=True).input_ids.cuda()
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = {**model_params, 'streamer': streamer, 'input_ids': input_ids}
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield the generated text as it becomes available. 
        id = str(uuid4())       
        for chunk in streamer:
            yield self.parse_chunk_output(
                id=id,
                output=chunk
            )
        yield self.parse_chunk_output(
            id=id, 
            output=chunk, 
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
                model=self.gai_config["model_name"],
                object="chat.completion.chunk",
                system_fingerprint=None,
                usage=None
                )
            return response
        except Exception as e:
            logger.error(f"TransformersEngine: error={e} id={id} output={output} finish_reason={finish_reason}")
            raise Exception(e)


    def create(self,messages,**model_params):
        self.prompt=self._apply_template(messages)
        if not self.tokenizer:
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