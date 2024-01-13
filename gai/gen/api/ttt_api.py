import os

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import asyncio
import os,json,io
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
generator = Gaigen.GetInstance()
# Pre-load default model
generator.load("mistral7b-exllama")

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
    try:
        model = request.model
        messages = request.messages
        model_params = request.model_dump(exclude={"model", "messages","stream"})  
        stream = request.stream
        gen = Gaigen.GetInstance().load(model)
        if stream:
            return StreamingResponse(json.dumps(jsonable_encoder(chunk))+"\n" for chunk in gen.create(
                model=model,
                messages=[message.model_dump() for message in messages],
                stream=True,
                **model_params
            ))
        else:
            return gen.create(
                model=model,
                messages=[message.model_dump() for message in messages],
                stream=False,
                **model_params
            )

    except Exception as e:
        logger.error(f"_create: error={e}")
        return JSONResponse(
            content={"_create: error=": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12031)