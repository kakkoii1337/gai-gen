from fastapi import FastAPI
import os
from dotenv import load_dotenv
load_dotenv()
from gai.common.logging import getLogger
logger = getLogger(__name__)
import asyncio
from gai.common.utils import this_dir

def app_version():
    ver_file=os.path.join(this_dir(__file__),"VERSION")
    if os.path.exists(ver_file):
        with open(ver_file) as f:
            return f.read()
    return "Not found."
APP_VERSION=app_version()

def lib_version():
    import subprocess
    import re
    try:
        command_output = subprocess.check_output("pip list | grep gai-lib-gen", shell=True).decode()
    except subprocess.CalledProcessError:
        raise Exception("gai-lib-gen is not installed.")
    version = re.search(r'(\d+\.\d+)', command_output)
    if version:
        return version.group()
    else:
        return "Not installed."
LIB_VERSION=lib_version()

def configure_logging():
    from gai.common.logging import configure_loglevel
    configure_loglevel()

def get_swagger_url():
    swagger_url=None
    if "SWAGGER_URL" in os.environ and os.environ["SWAGGER_URL"]:
        swagger_url=os.environ["SWAGGER_URL"]
        logger.info(f"swagger={swagger_url}")
    else:
        logger.info("swagger_url=disabled.")
    return swagger_url

def configure_cors(app: FastAPI):
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

def configure_semaphore():
    use_semaphore = os.getenv("USE_SEMAPHORE", "False").lower() == "true"
    semaphore = None
    if use_semaphore:
        logger.info("Using semaphore")
        import asyncio
        semaphore = asyncio.Semaphore(1)
    return semaphore

async def acquire_semaphore(semaphore):
    while semaphore:
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            logger.warn("_streaming: Server is busy")
            await asyncio.sleep(1)

def release_semaphore(semaphore):
    if semaphore:
        semaphore.release()
