from gai.common.generators_utils import chat_string_to_list
from gai.gen import Gaigen
import json
g = Gaigen.GetInstance().load("mistral7b-exllama")
with open("./tests/gen/ttt/tools/tools.txt","r") as f:
    tools = json.load(f)

# # This call does not involve tools
# with open("./tests/gen/ttt/tools/short_text.txt","r") as f:
#         context = f.read()
# messages = chat_string_to_list(context)
# print("\nText Prompt")
# response = g.create(messages=messages,stream=True,max_new_tokens=1000)
# for chunk in response:
#     print(chunk.choices[0].delta.content,end="",flush=True)


# #This call should not invoke the tools
# with open("./tests/gen/ttt/tools/short_text.txt","r") as f:
#         context = f.read()
# messages = chat_string_to_list(context)
# print("\nTools Prompt with Text")
# response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
# for chunk in response:
#     print(chunk.choices[0].delta.content,end="",flush=True)

# This call should invoke the tools
with open("./tests/gen/ttt/tools/long_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)
print("\nTools Prompt with Tools")
response = g.create(messages=messages,stream=True,max_new_tokens=1000,tools=tools)
for chunk in response:
    if chunk.choices[0].delta.content:
        print("CONTENT")
        print(chunk.choices[0].delta.content)
    if chunk.choices[0].delta.tool_calls[0].function.name:
        print("NAME")
        print(chunk.choices[0].delta.tool_calls[0].function.name)
    if chunk.choices[0].delta.tool_calls[0].function.arguments:
        print("ARGUMENTS")
        print(chunk.choices[0].delta.tool_calls[0].function.arguments)
