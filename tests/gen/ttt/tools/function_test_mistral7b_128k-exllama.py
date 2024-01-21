from gai.common.generators_utils import chat_string_to_list
from gai.gen import Gaigen
import json
g = Gaigen.GetInstance().load("mistral7b_128k-exllama")
with open("./tests/gen/ttt/tools/tools.txt","r") as f:
        tools = json.load(f)

# This call should not invoke the tools
with open("./tests/gen/ttt/tools/short_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)
print("\nshort context")
response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
for chunk in response:
    print(chunk.choices[0].delta.content,end="",flush=True)

# This call should invoke the tools
with open("./tests/gen/ttt/tools/long_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)
print("\nlong context")
response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
for chunk in response:
    print(chunk.choices[0].delta.content,end="",flush=True)