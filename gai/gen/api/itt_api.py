from fastapi import FastAPI, Body, UploadFile,File
from pydantic import BaseModel, Extra
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os,json,io
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version="0.0.1",
    docs_url=swagger_url
    )
dependencies.configure_cors(app)

# Enforce Thread-Safety
import asyncio
semaphore = asyncio.Semaphore(1)

from gai.gen import Gaigen
generator = Gaigen.GetInstance()

### ----------------- ITT ----------------- ###
class ImageToTextRequest(BaseModel):
    model: Optional[str] = "llava-transformers"
    messages: List
    stream: Optional[bool] = False
    class Config:
        extra = 'allow'  # Allow extra fields

@app.post("/gen/v1/vision/completions")
async def _image_to_text(request: ImageToTextRequest = Body(...)):
    gen = Gaigen.GetInstance().load(request.model)
    params = request.dict(exclude={"model", "messages","stream"})  # Get extra fields
    response = gen.create(
        messages=request.messages,
        stream=request.stream,
        **params
        )
    return response

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=12031, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=12031)