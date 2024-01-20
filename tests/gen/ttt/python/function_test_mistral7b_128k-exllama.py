from gai.common.generators_utils import chat_string_to_list
from gai.gen import Gaigen
g = Gaigen.GetInstance().load("mistral7b_128k-exllama")

with open("./tests/gen/ttt/python/short_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)

print("\nshort context")
response = g.create(messages=messages,stream=True,max_new_tokens=1000)
for chunk in response:
    print(chunk.choices[0].delta.content,end="",flush=True)

with open("./tests/gen/ttt/python/long_text.txt","r") as f:
        context = f.read()
messages = chat_string_to_list(context)

print("\nlong context")
response = g.create(messages=messages,stream=True,max_new_tokens=1000)
for chunk in response:
    print(chunk.choices[0].delta.content,end="",flush=True)
