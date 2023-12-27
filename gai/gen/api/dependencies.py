from fastapi import FastAPI
import os
from dotenv import load_dotenv
load_dotenv()

def configure_logging():
    from gai.common.logging import configure_loglevel
    configure_loglevel()

def get_swagger_url():
    from gai.common.logging import getLogger
    logger = getLogger(__name__)
    swagger_url=None
    if "SWAGGER_URL" in os.environ and os.environ["SWAGGER_URL"]:
        swagger_url=os.environ["SWAGGER_URL"]
        logger.info(f"swagger={swagger_url}")
    else:
        logger.info("swagger_url=disabled.")
    return swagger_url

def configure_cors(app: FastAPI):
    from gai.common.logging import getLogger
    logger = getLogger(__name__)
    from fastapi.middleware.cors import CORSMiddleware
    allowed_origins_str = "*"
    if "CORS_ALLOWED" in os.environ:
        allowed_origins_str = os.environ["CORS_ALLOWED"]    # from .env
    allowed_origins = allowed_origins_str.split(",")  # split the string into a list
    logger.info(f"allow_origins={allowed_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
