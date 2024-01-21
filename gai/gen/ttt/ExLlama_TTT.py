import torch, gc, re,os
from gai.common import logging, generators_utils
from gai.common.utils import this_dir, get_config_path
from gai.common.generators_utils import chat_string_to_list, has_ai_placeholder
logger = logging.getLogger(__name__)
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator as ExLlamaGen
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from uuid import uuid4
from datetime import datetime
from typing import List
import json
import re

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
        "stream",
        "tools",
        "tool_choice"
        ]

    def __init__(self,gai_config):
        if (gai_config is None):
            raise Exception("ExLlama_TTT: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("ExLlama_TTT: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("ExLlama_TTT: model_basename is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_config_path(), gai_config["model_path"], gai_config["model_basename"])+".safetensors"
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None

    def load(self):
        self.unload()
        logger.info(f"ExLlama_TTT.load: Loading model from {self.model_filepath}")

        # model
        model_dir=os.path.join(get_config_path(),self.gai_config["model_path"])
        if not os.path.exists(model_dir):
            raise Exception("ExLlama_TTT: model_dir is not found")
        model_config_path = os.path.join(model_dir, 'config.json')

        exllama_config = ExLlamaConfig(model_config_path)        
        exllama_config.max_seq_len = self.gai_config["max_seq_len"]
        exllama_config.model_path = self.model_filepath
        self.model = ExLlama(exllama_config)

        # tokenizer
        tokenizer_path= os.path.join(model_dir, 'tokenizer.model')
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
            raise Exception("ExLlama_TTT: tokenizer is not loaded")
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
        logger.debug(f"ExLlama_TTT._generate_simple: prompt={prompt}")
        
        max_seq_len = self.gai_config["max_seq_len"]
        self.client.end_beam_search()
        ids, mask = self.client.tokenizer.encode(prompt, return_mask = True, max_seq_len = max_seq_len)

        try:
            self.client.gen_begin(ids, mask = mask)
        except RuntimeError as e:
            if ((str(e).find("exceeds dimension size") != -1)):
                raise Exception("context_length_exceeded")
            raise e

        max_new_tokens = min(max_new_tokens, max_seq_len - ids.shape[1])

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
        logger.debug(f"ExLlama_TTT.generate: prompt={prompt}")

        # Map "max_tokens" to "max_new_tokens" to be compatible with OpenAI's API. We do not want to filter this off.
        if "max_tokens" in model_params and model_params["max_tokens"] is not None:
            model_params["max_new_tokens"]=model_params.pop("max_tokens")

        # Temperature approach 0 but cannot be 0
        if "temperature" in model_params and model_params["temperature"]==0:
            model_params["temperature"]=10e-10

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"],**model_params}
        logger.debug(f"ExLlama_TTT.generate: model_params={model_params}")
        
        input_count=self.token_count(prompt)
        logger.debug(f"ExLlama_TTT.generate: input token count={input_count}")

        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200

        response = self._generate_simple(prompt,max_new_tokens=max_new_tokens)
        
        logger.debug(f"ExLlama_TTT.generate: raw output={response}")
        
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
            model=self.gai_config["model_name"],
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,total_tokens=total_tokens)
            )
        return response    

    def _should_stop(self,new_text):
        stop_words=self.gai_config.get("stopping_words")
        stop_words.append("\"\n}")
        for stop_word in stop_words:
            if re.search(stop_word+"$",new_text):
                logger.debug(f"ExLlama_TTT._should_stop: stopped by : '{stop_word}'")        
                return True
        return False

    # TODO: To be used in future.
    def _check_response_type(self,prompt,**model_params):
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200        
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            TOOLS_TYPE_PREFIX_RE = r'\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
            tools_type_prefix = re.search(TOOLS_TYPE_PREFIX_RE,new_text)            
            if tools_type_prefix:
                return "tools"

            #TEXT_TYPE_PREFIX = " {\n    \"type\": \"text\",\n    \"text\": \""
            #TEXT_TYPE_PREFIX_RE = r'\s*{\s*(\")?type(\")?\s*:\s*"text",\s*(\")?text(\")?\s*:\s*"'
            TEXT_TYPE_PREFIX_RE = r'\s*[^{]'
            text_type_prefix = re.search(TEXT_TYPE_PREFIX_RE,new_text)
            if text_type_prefix:
                return "text"

    def _streaming(self,prompt,**model_params):

        stopping_words = self.gai_config["stopping_words"]

        new_text = ""
        last_text = ""

        logger.debug(f"ExLlama_TTT.streaming: prompt={prompt}")
        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"],**model_params}

        logger.debug(f"model_params: {model_params}")

        input_count=self.token_count(prompt)
        logger.debug(f"ExLlama_TTT.streaming: input token count={input_count}")

        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200

        self.client.end_beam_search()
        ids = self.tokenizer.encode(prompt)
        self.client.gen_begin_reuse(ids)
        id = str(uuid4())
        idx=0
        buffer=[]
        text_type_prefix_len = 0
        tools_type_prefix_len = 0
        prompt_len = len(prompt)

        response_type=None

        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            TOOLS_TYPE_PREFIX_RE = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
            tools_type_prefix = re.search(TOOLS_TYPE_PREFIX_RE,new_text)            
            if tools_type_prefix and (response_type=="tools" or response_type is None):
                if response_type is None:
                    response_type="tools"
                if tools_type_prefix_len == 0:
                    tools_type_prefix_len = len(tools_type_prefix.string)
                new_text = text[prompt_len+tools_type_prefix_len:]

                # Get new decoded token by taking difference from last response.
                # This is equivalent to new_token = self.tokenizer.decode(token) but faster.
                new_token = new_text.replace(last_text, "")

                # stop by natural end of sentence
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    buffer_str="".join(buffer)

                    JSON_SUFFIX_RE = r'\s*"\s*}\s*$'
                    buffer_str=re.sub(JSON_SUFFIX_RE, '',buffer_str)
                    yield self.parse_tools_output(id,name="",arguments=buffer_str,finish_reason="stop")
                    self.client.end_beam_search() 
                    return self.parse_tools_output(id,name="",arguments="",finish_reason="stop")

                # Add new token to a 10 token buffer:
                if len(buffer) < 10:
                    buffer.append(new_token)
                else:
                    # Remove oldest token from buffer and add new token
                    output_token = buffer[0]
                    yield self.parse_tools_output(id,name="",arguments=output_token)
                    buffer = buffer[1:]
                    buffer.append(new_token)

                # Stop by stopping words
                for stop_word in stopping_words:
                    buffer_str="".join(buffer)
                    if buffer_str.endswith(stop_word):
                        logger.debug(f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")
                        buffer_str=buffer_str.replace(stop_word,"")
                        yield self.parse_tools_output(id,name="",arguments=buffer_str)
                        self.client.end_beam_search() 
                        return self.parse_tools_output(id,name="",arguments="",finish_reason="stop")

                # Stop by max_new_tokens
                if i == max_new_tokens - 1 - len(buffer):
                    logger.debug(f"ExLlama_TTT.streaming: stopped by max_new_tokens: {max_new_tokens}")
                    # Yield all tokens in buffer
                    buffer_str="".join(buffer)
                    yield self.parse_tools_output(id,name="",arguments=buffer_str)
                    self.client.end_beam_search() 
                    return self.parse_tools_output(id,name="",arguments="", finish_reason="length")

            #TEXT_TYPE_PREFIX = " {\n    \"type\": \"text\",\n    \"text\": \""
            #TEXT_TYPE_PREFIX_RE = r'\s*{\s*(\")?type(\")?\s*:\s*"text",\s*(\")?text(\")?\s*:\s*"'
            TEXT_TYPE_PREFIX_RE = r'^\s*[^{\s]'
            text_type_prefix = re.search(TEXT_TYPE_PREFIX_RE,new_text)

            if text_type_prefix and (response_type=="text" or response_type is None):
                if response_type is None:
                    response_type="text"
                    yield self.parse_chunk_output(id,text_type_prefix.string)
                if text_type_prefix_len == 0:
                    text_type_prefix_len = len(text_type_prefix.string)
                new_text = text[prompt_len+text_type_prefix_len:]

                # Get new decoded token by taking difference from last response.
                # This is equivalent to new_token = self.tokenizer.decode(token) but faster.
                new_token = new_text.replace(last_text, "")

                # stop by natural end of sentence
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    buffer_str="".join(buffer)

                    JSON_SUFFIX_RE = r'\s*"\s*}\s*$'
                    buffer_str=re.sub(JSON_SUFFIX_RE, '',buffer_str)
                    yield self.parse_chunk_output(id,buffer_str)
                    self.client.end_beam_search() 
                    return self.parse_chunk_output(id,"", "stop")

                # Add new token to a 10 token buffer:
                if len(buffer) < 10:
                    buffer.append(new_token)
                else:
                    # Remove oldest token from buffer and add new token
                    output_token = buffer[0]
                    yield self.parse_chunk_output(id,output_token)
                    buffer = buffer[1:]
                    buffer.append(new_token)

                # Stop by stopping words
                for stop_word in stopping_words:
                    buffer_str="".join(buffer)
                    if buffer_str.endswith(stop_word):
                        logger.debug(f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")
                        buffer_str=buffer_str.replace(stop_word,"")
                        yield self.parse_chunk_output(id,buffer_str)
                        self.client.end_beam_search() 
                        return self.parse_chunk_output(id,"", "stop")

                # Stop by max_new_tokens
                if i == max_new_tokens - 1 - len(buffer):
                    logger.debug(f"ExLlama_TTT.streaming: stopped by max_new_tokens: {max_new_tokens}")
                    # Yield all tokens in buffer
                    buffer_str="".join(buffer)
                    yield self.parse_chunk_output(id,buffer_str)
                    self.client.end_beam_search() 
                    return self.parse_chunk_output(id,"", "length")

                # Update last_text so that it can be used to derive new_token next round
                last_text = new_text

        # Update last_text so that it can be used to derive new_token next round
        last_text = new_text

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
            model=self.gai_config["model_name"],
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None
            )
        return response


    # Sample:
    #ChatCompletionChunk(
    #     id='chatcmpl-8YqIOmXu1WLlYcYeMbhPg6yYWBQ1u', 
    #     choices=[
    #          Choice(delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[
    #              ChoiceDeltaToolCall(
    #                   id='toolcall-8YqIOmXu1WLlYcYeMbhPg6yYWBQ1u',
    #                   index=0,
    #                   type='function',
    #                   function=ChoiceDeltaToolCallFunction(
    #                       name='get_weather', 
    #                       arguments=''
    #                       )
    #                   )]),
    #          ]), 
    #          finish_reason=None, 
    #          index=0, 
    #          logprobs=None)],
    #     created=1703314868,
    #     model='gpt-4-0613',
    #     object='chat.completion.chunk',
    #     system_fingerprint=None)
    #.....
    def parse_tools_output(self, id, name, arguments,finish_reason=None):
        response = ChatCompletionChunk(
                id=id,                
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[
                            ChoiceDeltaToolCall(
                                id=id,
                                index=0,
                                type='function',
                                function=ChoiceDeltaToolCallFunction(
                                    name=name, 
                                    arguments=arguments
                                )
                            )
                        ]),
                        # "stop","length","content_filter"
                        finish_reason=finish_reason,
                        index=0,
                        logprobs=None,
                        message=None
                    )
                ],
                created=int(datetime.now().timestamp()),
                model=self.gai_config["model_name"],
                object="chat.completion.chunk",
                system_fingerprint=None,
                usage=None
            )
        
        return response

    def _apply_template(self, prompt:List):
        prompt = generators_utils.chat_list_to_string(prompt)
        return prompt

    def _remove_template(self, output:str):
        output = re.split('\n.+:',output)[-1].strip()
        return output

    def _apply_tools_message(self, messages:List,**model_params):
        # Check if tools are required
        if "tools" in model_params and model_params["tools"] is not None:
            tools = json.dumps(model_params["tools"], indent=4)
            tool_choice = "auto"
            if "tool_choice" in model_params and model_params["tool_choice"] is not None:
                tool_choice = model_params["tool_choice"]
            if tool_choice == "auto":
                prompt_file = "tools_prompt_auto.txt"
            with open(os.path.join(this_dir(__file__),prompt_file), "r") as f:
                tools_prompt = f.read()
            system_message = chat_string_to_list(tools_prompt)[0]

            # Somehow, removing the tools indentation fixed the issue of the tools not being recognized.
            tools_json=json.loads(tools)
            tools = json.dumps(tools_json)
            tools = tools.replace("{","{ ").replace("}"," }")

            system_message["content"] = system_message["content"].format(tools=tools)
        else:
            return messages

        ai_placeholder=None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

        return messages

    def create(self,messages,**model_params):
        messages = self._apply_tools_message(messages,**model_params)
        self.prompt=self._apply_template(messages)

        if not self.prompt:
            raise Exception("Exllama_TTT: prompt is required")

        if not self.client:
            self.load()

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"],**model_params}
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

        