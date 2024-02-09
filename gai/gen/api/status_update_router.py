import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from gai.gen.api.globals import status_updater
from gai.common.logging import logging
logger = logging.getLogger(__name__)

status_update_router = APIRouter()


# Used by websocket server to send keep-alive pings by data frame since control frame pings are not working


async def send_pings(websocket: WebSocket):
    while True:
        try:
            # Sleep time between pings, adjust as necessary
            await asyncio.sleep(10)
            # await websocket.send_text("ping")
        except asyncio.CancelledError:
            # If the task gets cancelled, break the loop
            break
        except Exception as e:
            # Handle exceptions, log them or break if necessary
            logger.error(f"Error sending ping: {e}")
            break


async def receive_text(websocket: WebSocket):
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
            logger.debug(f"Message text was: {data}")
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        logger.info("Client disconnected normally.")
        # Handle any cleanup or post-disconnect actions here
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Handle other exceptions that could occur


@status_update_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("rag.websocket: connecting.")

    await websocket.accept()
    logger.info("rag.websocket: connection accepted.")

    # Keep track of the websocket in a global variable for later use
    await status_updater.connect(websocket)

    # Start a task to echo messages
    receiver_task = asyncio.create_task(receive_text(websocket))

    # Run both tasks concurrently
    done, pending = await asyncio.wait(
        [receiver_task], return_when=asyncio.FIRST_COMPLETED
    )

    # If any task is done, cancel the other one
    for task in pending:
        task.cancel()

