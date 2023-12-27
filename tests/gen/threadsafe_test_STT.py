from threading import Thread
from gai.gen import Gaigen
import time
import requests

def worker():
    print("Inferencing...")
    start_time = time.time()

    response = requests.post("http://localhost:12031/gen/v1/audio/transcriptions", json={
        "model": "whisper-transformers"},files = {
    'file': open('../today-is-a-wonderful-day.wav', 'rb'),
    })
    end_time = time.time()
    # print(response.content)
    print("Time taken: {} seconds".format(end_time - start_time))    

thread_count = 5
threads = []

for i in range(thread_count):
    t = Thread(target=worker)
    t.start()
    threads.append(t)

for i in range(thread_count):
    threads[i].join()


