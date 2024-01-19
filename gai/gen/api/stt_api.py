from fastapi import FastAPI, UploadFile,File
from dotenv import load_dotenv
from gai.gen.api.errors import *
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_lib installed = {dependencies.LIB_VERSION}")
swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
    )
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()

from gai.gen import Gaigen
gen = Gaigen.GetInstance()

# Pre-load default model
def preload_model():
    try:
        # RAG does not use "default" model
        gen.load("whisper-transformers")
    except Exception as e:
        logger.error(f"Failed to preload default model: {e}")
        raise e
preload_model()

### ----------------- STT ----------------- ###
from io import BytesIO
from pydub import AudioSegment
import tempfile
from pathlib import Path
from fastapi import Form, File, UploadFile
import numpy as np

@app.post("/gen/v1/audio/transcriptions")
async def _speech_to_text(model: str = Form("whisper-transformers"),file: UploadFile = File(...)):
    try:
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

    except Exception as e:
        return InternalError(str(e))

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=12031, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=12031)