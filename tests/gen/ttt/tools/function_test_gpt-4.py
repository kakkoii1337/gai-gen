from gai.common.generators_utils import chat_string_to_list
from gai.gen import Gaigen
import json
g = Gaigen.GetInstance().load("gpt-4")
with open("./tests/gen/ttt/tools/tools.txt","r") as f:
        tools = json.load(f)

# This call should not invoke the tools
# with open("./tests/gen/ttt/tools/short_text.txt","r") as f:
#         context = f.read()
# messages = chat_string_to_list(context)
# print("\nshort context")
# response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
# for chunk in response:
#     print(chunk.choices[0].delta.content,end="",flush=True)

# This call should invoke the tools
with open("./tests/gen/ttt/tools/long_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)
print("\nlong context")
response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content,end="",flush=True)
    if chunk.choices[0].delta.tool_calls:
        print(chunk.choices[0].delta.tool_calls,end="",flush=True)

#        print(chunk.choices[0].delta,end="",flush=True)
#        if chunk.choices[0].delta.tool_calls:
#                 if chunk.choices[0].delta.tool_calls[0].function.name:
#                         print(chunk.choices[0].delta.tool_calls[0].function.name,end="",flush=True)
#                 elif chunk.choices[0].delta.tool_calls[0].function.arguments:
#                         print(chunk.choices[0].delta.tool_calls[0].function.arguments,end="",flush=True)
#                 else:
#                         print(chunk.choices[0].delta.tool_calls[0].function,end="",flush=True)
