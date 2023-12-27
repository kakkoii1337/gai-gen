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

### ----------------- TTT ----------------- ###
class MessageRequest(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[MessageRequest]
    class Config:
        extra = 'allow'  # Allow extra fields
    
@app.post("/gen/v1/chat/completions")
async def _text_to_text(request: ChatCompletionRequest = Body(...)):
    try:
        model = request.model
        messages = request.messages
        logger.debug(f"_create: model={model} messages={messages}")
        model_params = request.dict(exclude={"model", "messages"})  
        logger.debug(f"_create: model_params={model_params}")
        gen = Gaigen.GetInstance().load(model)

        stream = model_params.pop("stream", False)
        if stream:
            return StreamingResponse(json.dumps(jsonable_encoder(chunk)) for chunk in gen.create(
                model=model,
                messages=[message.dict() for message in messages],
                stream=True,
                **model_params
            ))
        else:
            return gen.create(
                model=model,
                messages=[message.dict() for message in messages],
                stream=False,
                **model_params
            )

    except Exception as e:
        logger.error(f"_create: error={e}")
        return JSONResponse(
            content={"_create: error=": str(e)},
            status_code=500
        )

### ----------------- STT ----------------- ###
from io import BytesIO
from pydub import AudioSegment
import tempfile
from pathlib import Path
from fastapi import Form, File, UploadFile
import numpy as np

@app.post("/gen/v1/audio/transcriptions")
async def _speech_to_text(model: str = Form("local-whisper"),file: UploadFile = File(...)):
    gen = Gaigen.GetInstance().load(model)
    print(f"Received file with filename: {file.filename} {file.content_type}")
    content = await file.read()
    
    # Convert webm file to wav if necessary
    if file.content_type == "audio/webm":
        audio = BytesIO(content)
        audio = AudioSegment.from_file(audio, format="webm")
        
        # Export audio to wav and get data
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            audio.export(tmp.name, format="wav")
            
            # Read wav file data into numpy array
            wav_file_data = np.memmap(tmp.name, dtype='h', mode='r')
        
    else:
        # If file is already in wav format, just read the data into numpy array
        #wav_file_data = np.frombuffer(content, dtype='h')
        wav_file_data = content

    return gen.create(file=wav_file_data)

### ----------------- TTS ----------------- ###
class TextToSpeechRequest(BaseModel):
    model: Optional[str] = "xtts-2"
    input: str
    voice: str = None
    language: str = None
    stream: Optional[bool] = False

@app.post("/gen/v1/audio/speech")
async def _text_to_speech(request: TextToSpeechRequest = Body(...)):
    gen = Gaigen.GetInstance().load(request.model)
    response = gen.create(
        voice=request.voice,
        input=request.input
        )
    return StreamingResponse(io.BytesIO(response), media_type="audio/mpeg")

### ----------------- ITT ----------------- ###
class ImageToTextRequest(BaseModel):
    model: Optional[str] = "llava-1.5"
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