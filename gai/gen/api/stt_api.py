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

### ----------------- STT ----------------- ###
from io import BytesIO
from pydub import AudioSegment
import tempfile
from pathlib import Path
from fastapi import Form, File, UploadFile
import numpy as np

@app.post("/gen/v1/audio/transcriptions")
async def _speech_to_text(model: str = Form("whisper-transformers"),file: UploadFile = File(...)):
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

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=12031, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=12031)