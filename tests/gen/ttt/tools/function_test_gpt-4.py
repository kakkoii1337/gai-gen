from print_response import print_response
from gai.common.generators_utils import chat_string_to_list
from gai.gen import Gaigen
import json
g = Gaigen.GetInstance().load("gpt-4")
with open("./tests/gen/ttt/tools/tools.txt", "r") as f:
    tools = json.load(f)

context = """
system:

        You are a helpful assistant tasked to assist the User in completing various tasks such as answering questions, 
        providing recommendations, helping with decision making, and more. You must always do what the user tells you to. 
        You will do your best to give the User the most accurate and relevant information. 
        If you do not know the answer, just say I don't know. Do not make up an answer.

user:

        Tell me the latest news on Singapore

assistant:"""
messages = chat_string_to_list(context)
print("\n>Should invoke Tools")
response = g.create(messages=messages, stream=True,
                    max_new_tokens=1000, tools=tools)

print_response(response)

context = """
system:

        You are a helpful assistant tasked to assist the User in completing various tasks such as answering questions, 
        providing recommendations, helping with decision making, and more. You must always do what the user tells you to. 
        You will do your best to give the User the most accurate and relevant information. 
        If you do not know the answer, just say I don't know. Do not make up an answer.

user:

        Tell me a one paragraph story

assistant:"""
messages = chat_string_to_list(context)
print("\n>Should not invoke Tools")
response = g.create(messages=messages, stream=True,
                    max_new_tokens=1000, tools=tools)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
