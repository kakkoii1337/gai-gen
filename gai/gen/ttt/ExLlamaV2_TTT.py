import torch, gc, re,os
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.common.utils import get_config_path,this_dir
from gai.common.generators_utils import chat_string_to_list, has_ai_placeholder
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator as ExLlamaGen
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from gai.gen.ttt.ExLlama_TTT import ExLlama_TTT
from uuid import uuid4
from datetime import datetime
from typing import List
import json

class ExLlamaV2_TTT(ExLlama_TTT):

    def _apply_template(self, prompt:List):
        prompt=generators_utils.chat_list_to_string(prompt)
        #prompt = generators_utils.chat_list_to_INST(prompt)
        return prompt

    def _remove_template(self, output:str):
        output=generators_utils.ASSISTANT_output_to_output(output)
        #output = generators_utils.INST_output_to_output(output)
        return output



        