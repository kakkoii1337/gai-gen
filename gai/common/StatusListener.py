import websockets
import json


class StatusListener:

    def __init__(self, uri):
        self.uri = uri

    async def listen(self, callback=None):
        async with websockets.connect(self.uri) as websocket:
            print(f"Connected to {self.uri}")
            try:
                while True:
                    message = await websocket.recv()
                    status_update = json.loads(message)
                    if callback:
                        callback(status_update)
                    print(f"Received status update: {status_update}")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection closed: {e}")
