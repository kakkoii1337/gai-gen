from threading import Thread
from gai.gen import Gaigen
import time
import requests
import subprocess
import tempfile

def worker():
    print("Inferencing...")
    start_time = time.time()

    response = requests.post("http://localhost:12031/gen/v1/audio/speech", json={
        "model": "xtts-2", 
        "input":"I think there is no direct bus. You can take 185 and change to MRT at buona vista. 185 should be arriving in 5 minutes.", 
        "stream":True
        })
    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))

    # Create a temporary file and write the response content to it
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
    #     tf.write(response.content)
    #     temp_file_name = tf.name
    # Use ffplay to play the audio file
    #subprocess.run(['ffplay', '-autoexit', '-nodisp', '-hide_banner', temp_file_name])

thread_count = 5
threads = []

for i in range(thread_count):
    t = Thread(target=worker)
    t.start()
    threads.append(t)

for i in range(thread_count):
    threads[i].join()


