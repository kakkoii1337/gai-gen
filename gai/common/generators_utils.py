import json
import os
from gai.common.utils import get_config

# A simple utility to validate if all items in model params are in the whitelist.
def validate_params(model_params,whitelist_params):
    for key in model_params:
        if key not in whitelist_params:
            raise Exception(f"Invalid param '{key}'. Valid params are: {whitelist_params}")

# A simple utility to filter items in model params that are also in the whitelist.
def filter_params(model_params,whitelist_params):
    filtered_params={}
    for key in model_params:
        if key in whitelist_params:
            filtered_params[key]=model_params[key]
    return filtered_params

# A simple utility to load generators config.
def load_generators_config():
    return get_config()["gen"]

# This is used to compress a list into a smaller string to be passed as a single USER message to the prompt template.
def chat_list_to_string(messages):
    if type(messages) is str:
        return messages
    prompt=""        
    for message in messages:
        if prompt:
            prompt+="\n"
        content = message['content'].strip()
        role = message['role'].strip()
        if content:
            prompt += f"{role}: {content}"
        else:
            prompt += f"{role}:"
    return prompt

async def word_streamer_async( char_generator):
    buffer = ""
    async for byte_chunk in char_generator:
        if type(byte_chunk) == bytes:
            byte_chunk = byte_chunk.decode("utf-8", "replace")
        buffer += byte_chunk
        words = buffer.split(" ")
        if len(words) > 1:
            for word in words[:-1]:
                yield word
                yield " "
            buffer = words[-1]
    yield buffer            

def word_streamer( char_generator):
    buffer = ""
    for chunk in char_generator:
        if chunk:
            if type(chunk) == bytes:
                chunk = chunk.decode("utf-8", "replace")
            buffer += chunk
            words = buffer.split(" ")
            if len(words) > 1:
                for word in words[:-1]:
                    yield word
                    yield " "
                buffer = words[-1]
    yield buffer
