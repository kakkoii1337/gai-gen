from fastapi import WebSocketDisconnect, WebSocket
from fastapi.websockets import WebSocketState
import asyncio
from gai.common.logging import logging
import json
logger = logging.getLogger(__name__)


class StatusUpdater:

    def __init__(self):
        self.status = None
        self.websocket = None

    async def connect(self, websocket: WebSocket):
        self.websocket = websocket

    async def disconnect(self, websocket: WebSocket):
        await websocket.close()
        self.websocket = None

    # update_progress is the same as update_status,
    # but it returns an integer between 0 to 100
    async def update_progress(self, i, max):
        self.status = int(i*100/max)
        if self.websocket is not None:
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.info("rag.index: websocket is disconnected.")
                return
            await asyncio.create_task(self.websocket.send_json({'status': self.status}))

    def get_status(self):
        return self.status
