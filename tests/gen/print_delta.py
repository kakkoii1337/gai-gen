import sys
import json

for line in sys.stdin:
    result = json.loads(line)
    content = result["choices"][0]["delta"]["content"]
    if content:
        print(content, end="", flush=True)
