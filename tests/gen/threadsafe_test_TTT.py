from threading import Thread
from gai.gen import Gaigen
import json
import time
import requests

def worker():
    print("Inferencing...")
    start_time = time.time()
    response = requests.post("http://localhost:12031/gen/v1/chat/completions", json={
        "model": "mistral7b-exllama",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "Tell me a one paragraph story."
            }
        ]
    })
    end_time = time.time()
    # for chunk in response.iter_lines():
    #     print(json.loads(chunk)["choices"][0]["delta"]["content"],end='',flush=True)
    print("Time taken: {} seconds".format(end_time - start_time))    


thread_count = 5
threads = []

for i in range(thread_count):
    t = Thread(target=worker)
    t.start()
    threads.append(t)

for i in range(thread_count):
    threads[i].join()