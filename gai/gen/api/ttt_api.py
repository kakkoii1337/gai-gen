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

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=12031, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=12031)