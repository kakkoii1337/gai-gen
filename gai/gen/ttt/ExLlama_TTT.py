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


    # The purpose of this function is to classify the nature of the text based on its initial characters.
    # The text can be classified either as "tool" or "text".
    # A Tool begins with '{"type":"function","function":'
    # A Text begins with '{"type":"text", "text":' OR any character that is not '{'
    # If neither of the above, then it is None and it is not classified yet.
    def classify_text_nature(self, text):

        # Look for the pattern that matches '{"type":"function","function":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
        if re.search(pattern, text):
            return "tools"

        # Look for the pattern that matches '{"type":"tool","tool":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"tool",\s*(\\n)?\s*(\")?tool(\")?\s*:\s*'
        if re.search(pattern, text):
            return "tools"
        
        # Look for the pattern that doesn't begin with '{'
        pattern = r'^\s*[^{\s]'
        if re.search(pattern, text):
            return "text"

        # Look for the pattern that matches '{"type":"text", "text":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"text",\s*(\\n)?\s*(\")?text(\")?\s*:\s*'
        if re.search(pattern, text):
            return "text"

        return None

    # If the response is a tool, the first yielded output will return
    # the tool name.
    def _yield_tool_name_output(self, text):
        tool_name_pattern = r'(\")?name(\")?\s*:\s*\"(.*?)\",'
        match = re.search(tool_name_pattern, text)
        if match:
            tool_name=match.group(3)
            logger.debug(
                f"ExLlama_TTT.streaming: tool_name={tool_name}")
            output = ChunkOutputBuilder.BuildToolHead(
                generator=self.gai_config["model_name"], 
                tool_name=tool_name)
            return output
        return None

    # If the response is a tool, the next yielded output will return
    # the tool arguments.
    def _yield_tool_arguments_output(self, text):
        tool_arguments_pattern = r'"parameters":(\s*\{[\s\S]*?\})'
        match = re.search(tool_arguments_pattern, text)
        if match:
            tool_arguments=match.group(1)
            logger.debug(
                f"ExLlama_TTT.streaming: tool_arguments={tool_arguments}")
            output = ChunkOutputBuilder.BuildToolBody(
                generator=self.gai_config["model_name"], 
                tool_arguments=tool_arguments)
            return output
        return None
    
    def _yield_tool_stop_output(self, finish_reason, stop_word=None):
        if finish_reason == "tool_calls":
            logger.debug(
                f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
        if finish_reason == "stop":
            logger.debug(
                f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")   
        if finish_reason == "length":
            logger.debug(
                f"ExLlama_TTT.streaming: stopped by : length")   
        self.client.end_beam_search()
        return ChunkOutputBuilder.BuildToolTail(
            generator=self.gai_config["model_name"], 
            finish_reason=finish_reason)

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
        tool_name_output = None
        tool_arguments_output = None

        initial_text = ''
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            # At this point, we cannot tell if the stream is returning a tool call or text response.
            # In order to do that, we will compare the text generated so far with the JSON pattern
            # corresponding to a function call. If it matches, then it is a tool call, otherwise it is a text response.
            # For example, a stream starting with {"type":"function","function": will be considered a match.
            response_type = self.classify_text_nature(new_text)

            if response_type == "tools":

                # initial text is the text that were accumulated when classification was still unknown.
                # Once the response_type is confirmed, the initial_text must be flushed out.
                if (initial_text is None):
                    initial_text = new_text

                # This is the generated text
                new_text = text[prompt_len+len(initial_text):]

                # This is the new sub-word decoded from the token
                new_token = new_text.replace(last_text, "")

                # Find tool name and yield output head
                if not tool_name_output:
                    tool_name_output = self._yield_tool_name_output(new_text)
                    if tool_name_output:
                        yield tool_name_output

                # Find tool args and yield output body
                if not tool_arguments_output:
                    tool_arguments_output = self._yield_tool_arguments_output(new_text)
                    if tool_arguments_output:
                        yield tool_arguments_output

                # stop by natural end of sentence and yield output tail
                if token.item() == self.tokenizer.eos_token_id:
                    yield self._yield_tool_stop_output("tool_calls")
                    return

                # Stop by stopping words. Exception case.
                for stop_word in stopping_words:
                    if new_text.endswith(stop_word):
                        yield self._yield_tool_stop_output("stop", stop_word)
                        return

                # Stop by max_new_tokens
                # Other than finish_reason="tool_calls", the output should be treated as exception
                if i == max_new_tokens - prompt_len - tools_type_prefix_len:
                    yield self._yield_tool_stop_output("length")
                    return

            if response_type == "text":

                # new_text = total text - prompt
                new_text = text[prompt_len:]

                # initial text is the text that were accumulated when classification was still unknown.
                # Once the response_type is confirmed, the initial_text must be flushed out.
                if (not initial_text):
                    initial_text = new_text
                    yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])

                # new_token = new_text - last_text
                # This is equivalent to new_token = self.tokenizer.decode(token) but faster.
                new_token = new_text.replace(last_text, "")

                # stop by natural end of sentence
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    buffer_str = "".join(buffer)

                    JSON_SUFFIX_RE = r'\s*"\s*}\s*$'
                    buffer_str = re.sub(JSON_SUFFIX_RE, '', buffer_str)

                    # Flush the buffer and stop
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                    
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")                    
                    self.client.end_beam_search()
                    return

                # Add new token to a 10 token holding buffer to monitor for stopping word.
                buffer.append(new_token)
                if len(buffer) == 11:
                    # Once the buffer overflows, the output is dequeued and yielded.
                    output_token = buffer[0]
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=output_token)                    
                    buffer = buffer[1:]

                # Stop by stopping words
                for stop_word in stopping_words:
                    buffer_str = "".join(buffer)
                    if buffer_str.endswith(stop_word):
                        logger.debug(
                            f"ExLlama_TTT.streaming: stopped by : '{stop_word}'")
                        buffer_str = buffer_str.replace(stop_word, "")

                        # Flush the buffer and stop
                        yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                        
                        yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")
                        self.client.end_beam_search()
                        return

                # Stop by max_new_tokens exclude buffer
                if i == max_new_tokens - 1 - len(buffer):
                    logger.debug(
                        f"ExLlama_TTT.streaming: stopped by max_new_tokens: {max_new_tokens}")
                    # Yield all tokens in buffer
                    buffer_str = "".join(buffer)

                    # Flush the buffer and stop
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="length")
                    self.client.end_beam_search()
                    return

                # Update last_text so that it can be used to derive new_token next round
                last_text = new_text

        # all done:
        self.client.end_beam_search()
        if response_type is None:
            raise Exception(f"ExLlama_TTT: Response type cannot be classified: {text[prompt_len:]}")
        return

    # It is just a wrapper around _streaming. It may not be the most efficient approach but since generating is seldom used in practise, we can afford to be less efficient.
    def _generating(self, prompt, **model_params):
        text = ""
        finish_reason = None
        chunk_type=None
        for chunk in self._streaming(prompt, **model_params):

            # That means this is a stream of tokens
            if chunk.choices[0].delta.content:
                if chunk_type is None:
                    chunk_type='text'
                text += chunk.choices[0].delta.content

            # That means this is a tool call. For tool calls, we will only yield tool name and tool arguments.
            if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.name:
                if chunk_type is None:
                    chunk_type='tool'
                function_name = chunk.choices[0].delta.tool_calls[0].function.name

            if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.arguments:
                function_arguments = chunk.choices[0].delta.tool_calls[0].function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason


        if chunk_type == 'text':
            return OutputBuilder.BuildContent(
                generator=self.gai_config["model_name"], 
                finish_reason=finish_reason, 
                content=text, 
                prompt_tokens=len(prompt), 
                new_tokens=len(text)
                )

        if chunk_type == 'tool':        
            return OutputBuilder.BuildTool(
                generator=self.gai_config["model_name"],
                function_name=function_name,
                function_arguments=function_arguments,
                prompt_tokens=len(prompt),
                new_tokens=len(text)
            )

        raise Exception('Unknown chunk type.')


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
