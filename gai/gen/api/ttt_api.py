import os
import subprocess
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import asyncio
import os,json,io

from gai.gen.api.errors import *
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_lib_gen installed = {dependencies.LIB_VERSION}")
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
from gai.common.utils import get_config_path, get_config
def preload_model():
    try:
        gai_config = get_config()
        if "default" in gai_config["gen"]:
            default_generator_name = gai_config["gen"]["default"]
            gen.load(default_generator_name)
    except Exception as e:
        return PreloadModelError(str(e))
preload_model()

### ----------------- TTT ----------------- ###
class MessageRequest(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mistral7b-exllama"
    messages: List[MessageRequest]
    stream: Optional[bool] = False
    class Config:
        extra = 'allow'  # Allow extra fields
    
@app.post("/gen/v1/chat/completions")
async def _text_to_text(request: ChatCompletionRequest = Body(...)):
    response=None
    try:
        model = request.model
        messages = request.messages
        model_params = request.model_dump(exclude={"model", "messages","stream"})  
        stream = request.stream
        if model != gen.generator_name:
            raise Exception("model_service_mismatch")

        response = gen.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            stream=stream,
            **model_params
        )
        if stream:
            return StreamingResponse(json.dumps(jsonable_encoder(chunk))+"\n" for chunk in response)
        else:
            return response
    except Exception as e:
        if (str(e)=='context_length_exceeded'):
            return ContextLengthExceededError()
        if (str(e)=='model_service_mismatch'):
            return ModelServiceMismatchError()
        return InternalError(str(e))

class ChatInstallRequest(BaseModel):
    repo:str
    model:str
    files:Optional[List[str]] = None
@app.put("/gen/v1/chat/install")
async def _chat_install(req:ChatInstallRequest):
    try:
        if req.files:
            files = " ".join(files)
        else:
            files = ""

        # construct command
        cmd = f"huggingface-cli download {req.repo}/{req.model} {files} --local-dir ~/gai/models/{req.model} --local-dir-use-symlinks False"
        
        # execute command
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(stderr.decode())
        
        return JSONResponse(status_code=200,content={
            "type":"success",
            "code":"installed",
            "message":"Model installed successfully."
        })
    except Exception as e:
        return InternalError(str(e))


class ChatDefaultConfigRequest(BaseModel):
    generator_name:str
    generator_config:Dict[str,Any]
@app.put("/gen/v1/chat/config")
async def _chat_default_config(req:ChatDefaultConfigRequest):
    try:
        # read
        config_path = get_config_path()
        with open(os.path.join(config_path,"gai.json"),'r') as f:
            config = json.load(f)
            
        config["gen"]["default"] = req.generator_name
        config["gen"][req.generator_name] = req.generator_config

        # write
        with open(os.path.join(config_path,"gai.json"),'w') as f:
            json.dump(config,f,indent=4)
    except Exception as e:
        return InternalError(str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12031)