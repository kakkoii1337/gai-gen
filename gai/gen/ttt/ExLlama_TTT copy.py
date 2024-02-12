from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder
from gai.gen.ttt.OutputBuilder import OutputBuilder
import re
import json
from typing import List
from datetime import datetime
from uuid import uuid4
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from exllama.generator import ExLlamaGenerator as ExLlamaGen
from exllama.tokenizer import ExLlamaTokenizer
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
import torch
import gc
import re
import os
from gai.common import logging, generators_utils
from gai.common.utils import this_dir, get_config_path
from gai.common.generators_utils import chat_string_to_list, has_ai_placeholder
logger = logging.getLogger(__name__)


class ExLlama_TTT:

    param_whitelist = [
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

    def __init__(self, gai_config):
        if (gai_config is None):
            raise Exception("ExLlama_TTT: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("ExLlama_TTT: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("ExLlama_TTT: model_basename is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_config_path(
        ), gai_config["model_path"], gai_config["model_basename"])+".safetensors"
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None

    def load(self):
        self.unload()
        logger.info(
            f"ExLlama_TTT.load: Loading model from {self.model_filepath}")

        # model
        model_dir = os.path.join(
            get_config_path(), self.gai_config["model_path"])
        if not os.path.exists(model_dir):
            raise Exception("ExLlama_TTT: model_dir is not found")
        model_config_path = os.path.join(model_dir, 'config.json')

        exllama_config = ExLlamaConfig(model_config_path)
        exllama_config.max_seq_len = self.gai_config["max_seq_len"]
        exllama_config.model_path = self.model_filepath
        self.model = ExLlama(exllama_config)

        # tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.model')
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)

        # generator
        self.client = ExLlamaGen(
            self.model, self.tokenizer, ExLlamaCache(self.model))

        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.client
            del self.prompt
        except:
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
        encoded = self.tokenizer.encode(text)
        return len(encoded.tolist()[0])

    def get_response(self, output, ai_role="ASSISTANT"):
        return re.split(rf'{ai_role}:', output, flags=re.IGNORECASE)[-1].strip().replace('\n\n', '\n').replace('</s>', '')

    def _init_settings(self, model_params):
        self.client.settings.temperature = model_params["temperature"] if "temperature" in model_params and model_params[
            "temperature"] is not None else self.client.settings.temperature
        self.client.settings.top_p = model_params["top_p"] if "top_p" in model_params and model_params[
            "top_p"] is not None else self.client.settings.top_p
        self.client.settings.min_p = model_params["min_p"] if "min_p" in model_params and model_params[
            "min_p"] is not None else self.client.settings.min_p
        self.client.settings.top_k = model_params["top_k"] if "top_k" in model_params and model_params[
            "top_k"] is not None else self.client.settings.top_k

        self.client.settings.token_repetition_penalty_max = model_params["token_repetition_penalty_max"] if "token_repetition_penalty_max" in model_params and model_params[
            "token_repetition_penalty_max"] is not None else self.client.settings.token_repetition_penalty_max
        self.client.settings.token_repetition_penalty_sustain = model_params["token_repetition_penalty_sustain"] if "token_repetition_penalty_sustain" in model_params and model_params[
            "token_repetition_penalty_sustain"] is not None else self.client.settings.token_repetition_penalty_sustain
        self.client.settings.token_repetition_penalty_decay = model_params["token_repetition_penalty_decay"] if "token_repetition_penalty_decay" in model_params and model_params[
            "token_repetition_penalty_decay"] is not None else self.client.settings.token_repetition_penalty_decay

        self.client.settings.typical = model_params["typical"] if "typical" in model_params and model_params[
            "typical"] is not None else self.client.settings.typical
        self.client.settings.beams = model_params["beams"] if "beams" in model_params and model_params[
            "beams"] is not None else self.client.settings.beams
        self.client.settings.beam_length = model_params["beam_length"] if "beam_length" in model_params and model_params[
            "beam_length"] is not None else self.client.settings.beam_length


    def _preprocessing(self,prompt,**model_params):
        # Map "max_tokens" to "max_new_tokens" to be compatible with OpenAI's API. We do not want to filter this off.
        if "max_tokens" in model_params and model_params["max_tokens"] is not None:
            model_params["max_new_tokens"] = model_params.pop("max_tokens")

        # Temperature approach 0 but cannot be 0
        if "temperature" in model_params and model_params["temperature"] == 0:
            model_params["temperature"] = 10e-10

        model_params = generators_utils.filter_params(
            model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"], **model_params}

        return model_params



    def _should_stop(self, new_text):
        stop_words = self.gai_config.get("stopping_words")
        stop_words.append("\"\n}")
        for stop_word in stop_words:
            if re.search(stop_word+"$", new_text):
                logger.debug(
                    f"ExLlama_TTT._should_stop: stopped by : '{stop_word}'")
                return True
        return False

    # TODO: To be used in future.
    def _check_response_type(self, prompt, **model_params):
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            TOOLS_TYPE_PREFIX_RE = r'\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
            tools_type_prefix = re.search(TOOLS_TYPE_PREFIX_RE, new_text)
            if tools_type_prefix:
                return "tools"

            # TEXT_TYPE_PREFIX = " {\n    \"type\": \"text\",\n    \"text\": \""
            # TEXT_TYPE_PREFIX_RE = r'\s*{\s*(\")?type(\")?\s*:\s*"text",\s*(\")?text(\")?\s*:\s*"'
            TEXT_TYPE_PREFIX_RE = r'\s*[^{]'
            text_type_prefix = re.search(TEXT_TYPE_PREFIX_RE, new_text)
            if text_type_prefix:
                return "text"

    def _streaming_text(self, text_type_prefix, response_type, **model_params):
        if response_type != "text":
            raise Exception("ExLlama_TTT.streaming_text: incorrect_response_type. Expecting type of be text.")

    def _streaming(self, prompt, **model_params):
        logger.debug(f"ExLlama_TTT.streaming: prompt={prompt}")

        model_params= self._preprocessing(prompt, **model_params)
        logger.debug(f"ExLlama_TTT.streaming: model_params={model_params}")

        input_count = self.token_count(prompt)
        logger.debug(f"ExLlama_TTT.streaming: input token count={input_count}")

        # Initialize exllama settings
        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200

        # ----- generating and streaming should be identical above this line -----

        stopping_words = self.gai_config["stopping_words"]
        new_text = ""
        last_text = ""

        self.client.end_beam_search()
        ids = self.tokenizer.encode(prompt)
        self.client.gen_begin_reuse(ids)
        id = str(uuid4())
        idx = 0
        buffer = []
        text_type_prefix_len = 0
        tools_type_prefix_len = 0
        prompt_len = len(prompt)

        response_type = None
        tool_name = None
        parameters = None

        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            # At this point, we cannot tell if the stream is returning a tool call or text response.
            # In order to do that, we will compare the text generated so far with the JSON pattern
            # corresponding to a function call. If it matches, then it is a tool call, otherwise it is a text response.
            # For example, a stream starting with {"type":"function","function": will be considered a match.
            TOOLS_TYPE_PREFIX_RE = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
            tools_type_prefix = re.search(TOOLS_TYPE_PREFIX_RE, new_text)
            if tools_type_prefix and (response_type == "tools" or response_type is None):

                # ------------------ When it reaches here, that means it is a tool call ------------------

                if response_type is None:
                    response_type = "tools"
                    tools_type_prefix_len = len(tools_type_prefix.string)

                # This is the generated text string
                new_text = text[prompt_len+tools_type_prefix_len:]

                # This is the new sub-word decoded from the token
                new_token = new_text.replace(last_text, "")

                if len(buffer) < max_new_tokens - i:
                    buffer.append(new_token)
                else:
                    raise Exception(
                        "ExLlama_TTT.streaming: Tool call error due to exceeded max_new_tokens.")

                # Find tool name and yield output start
                if not tool_name:
                    tool_name_re = r'(\")?name(\")?\s*:\s*\"(.*?)\",'
                    match = re.search(tool_name_re, new_text)
                    if match:
                        tool_name = match.group(3)
                        logger.debug(
                            f"ExLlama_TTT.streaming: tool_name={tool_name}")
                        # Yield Tool Output Start
                        output = ChunkOutputBuilder.BuildToolHead(
                            generator=self.gai_config["model_name"], 
                            tool_name=tool_name)
                        yield output

                # Find args and yield output body
                if not parameters:
                    match = re.search(
                        r'"parameters":(\s*\{[\s\S]*?\})', new_text)
                    if match:
                        parameters = match.group(1).strip()
                        output = ChunkOutputBuilder.BuildToolBody(
                            generator=self.gai_config["model_name"], 
                            tool_arguments=parameters)
                        yield output

                # stop by natural end of sentence
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    yield ChunkOutputBuilder.BuildToolTail(
                        generator=self.gai_config["model_name"], 
                        finish_reason="tool_calls")
                    self.client.end_beam_search()
                    return

                # Stop by stopping words
                for stop_word in stopping_words:
                    if new_text.endswith(stop_word):
                        logger.debug(
                            f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")
                        yield ChunkOutputBuilder.BuildToolTail(
                            generator=self.gai_config["model_name"], 
                            finish_reason="stop")
                        self.client.end_beam_search()
                        return

                # Stop by max_new_tokens
                if i == max_new_tokens - prompt_len - tools_type_prefix_len:
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by max_new_tokens: {max_new_tokens}")
                    output = ChunkOutputBuilder.BuildToolTail(
                        generator=self.gai_config["model_name"], 
                        finish_reason="length")
                    self.client.end_beam_search()
                    yield output
                    return

            # TEXT_TYPE_PREFIX_RE = string that does not begin with "{"
            TEXT_TYPE_PREFIX_RE = r'^\s*[^{\s]'
            text_type_prefix = re.search(TEXT_TYPE_PREFIX_RE, new_text)

            # TEXT_TYPE_PREFIX_RE = string that begin with '{ "type": "text",  "text":'
            TEXT_TYPE_PREFIX_RE_v2 = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"text",\s*(\\n)?\s*(\")?text(\")?\s*:\s*'
            text_type_prefix_v2 = re.search(TEXT_TYPE_PREFIX_RE_v2, new_text)

            # 
            if (text_type_prefix or text_type_prefix_v2) and (response_type == "text" or response_type is None):
                if response_type is None:
                    response_type = "text"
                    if text_type_prefix:
                        yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
                        yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=text_type_prefix.string)
                        #yield self.parse_chunk_output(id, text_type_prefix.string)
                    elif text_type_prefix_v2:
                        yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
                        yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=text_type_prefix_v2.string)
                        #yield self.parse_chunk_output(id, text_type_prefix_v2.string)

                # This is only required once in the beginning where a bunch of texts are accumulated while the response_type is still not confirmed.
                # Once the response_type is confirmed, this will not be required.
                if text_type_prefix_len == 0:
                    if text_type_prefix:
                        text_type_prefix_len = len(text_type_prefix.string)
                    elif text_type_prefix_v2:
                        text_type_prefix_len = len(text_type_prefix_v2.string)

                # new_text is needed to find new_token                
                new_text = text[prompt_len+text_type_prefix_len:]

                # Get new decoded token by taking difference from last response.
                # This is equivalent to new_token = self.tokenizer.decode(token) but faster.
                new_token = new_text.replace(last_text, "")

                # stop by natural end of sentence
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    buffer_str = "".join(buffer)

                    JSON_SUFFIX_RE = r'\s*"\s*}\s*$'
                    buffer_str = re.sub(JSON_SUFFIX_RE, '', buffer_str)
                    #yield self.parse_chunk_output(id, buffer_str)
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                    
                    #return self.parse_chunk_output(id, "", "stop")
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")                    
                    self.client.end_beam_search()
                    return

                # Add new token to a 10 token buffer:
                # The buffer holds the last 10 generated tokens to check for stopping word.
                if len(buffer) < 10:
                    buffer.append(new_token)
                else:
                    # Remove oldest token from buffer and add new token
                    output_token = buffer[0]
                    #yield self.parse_chunk_output(id, output_token)
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=output_token)                    
                    buffer = buffer[1:]
                    buffer.append(new_token)

                # Stop by stopping words
                for stop_word in stopping_words:
                    buffer_str = "".join(buffer)
                    if buffer_str.endswith(stop_word):
                        logger.debug(
                            f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")
                        buffer_str = buffer_str.replace(stop_word, "")
                        yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                        
                        #yield self.parse_chunk_output(id, buffer_str)
                        #return self.parse_chunk_output(id, "", "stop")
                        yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")
                        self.client.end_beam_search()
                        return

                # Stop by max_new_tokens
                if i == max_new_tokens - 1 - len(buffer):
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by max_new_tokens: {max_new_tokens}")
                    # Yield all tokens in buffer
                    buffer_str = "".join(buffer)
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)
                    #yield self.parse_chunk_output(id, buffer_str)
                    #return self.parse_chunk_output(id, "", "length")
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="length")
                    self.client.end_beam_search()
                    return

                # Update last_text so that it can be used to derive new_token next round
                last_text = new_text

        # Update last_text so that it can be used to derive new_token next round
        last_text = new_text

        # all done:
        self.client.end_beam_search()
        return

    # It is just a wrapper around _streaming. It may not be the most efficient approach but since generating is seldom used in practise, we can afford to be less efficient.
    def _generating(self, prompt, **model_params):
        text = ""
        finish_reason = None
        for chunk in self._streaming(prompt, **model_params):

            # That means this is a stream of tokens
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content

            # That means this is a tool call. For tool calls, we will only yield tool name and tool arguments.
            if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.name:
                function_name = chunk.choices[0].delta.tool_calls[0].function.name

            if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.arguments:
                function_arguments = chunk.choices[0].delta.tool_calls[0].function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        if text:
            return OutputBuilder.BuildContent(
                generator=self.gai_config["model_name"], 
                finish_reason=finish_reason, 
                content=text, 
                prompt_tokens=len(prompt), 
                new_tokens=len(text)
                )
        
        return OutputBuilder.BuildTool(
            generator=self.gai_config["model_name"],
            function_name=function_name,
            function_arguments=function_arguments,
            prompt_tokens=len(prompt),
            new_tokens=len(text)
        )

    def _apply_template(self, prompt: List):
        prompt = generators_utils.chat_list_to_string(prompt)
        return prompt

    def _remove_template(self, output: str):
        output = re.split('\n.+:', output)[-1].strip()
        return output

    def _apply_tools_message(self, messages: List, **model_params):
        # Check if tools are required
        if "tools" in model_params and model_params["tools"] is not None:
            tools = json.dumps(model_params["tools"], indent=4)
            tool_choice = "auto"
            if "tool_choice" in model_params and model_params["tool_choice"] is not None:
                tool_choice = model_params["tool_choice"]
            if tool_choice == "auto":
                prompt_file = "tools_prompt_auto.txt"
            with open(os.path.join(this_dir(__file__), prompt_file), "r") as f:
                tools_prompt = f.read()
            system_message = chat_string_to_list(tools_prompt)[0]

            # Somehow, removing the tools indentation fixed the issue of the tools not being recognized.
            tools_json = json.loads(tools)
            tools = json.dumps(tools_json)
            tools = re.sub(r'/s+', ' ', tools)
            tools = tools.replace("{", "{ ").replace("}", " }")
            try:
                system_message["content"] = system_message["content"].format(
                    tools=tools)
            except Exception as e:
                logger.error(
                    f"ExLlama_TTT._apply_tools_message: Error applying tools message: {e}")
                raise Exception(
                    "ExLlama_TTT._apply_tools_message: Error applying tools template.")
        else:
            return messages

        ai_placeholder = None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

        return messages

    def create(self, messages, **model_params):
        messages = self._apply_tools_message(messages, **model_params)
        self.prompt = self._apply_template(messages)

        if not self.prompt:
            raise Exception("Exllama_TTT: prompt is required")

        if not self.client:
            self.load()

        model_params = generators_utils.filter_params(
            model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"], **model_params}
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
