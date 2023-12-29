VERSION='0.15'
from setuptools import setup, find_packages
from os.path import abspath
import subprocess, os, sys
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='gai-lib-gen',
    version=VERSION,
    author="kakkoii1337",
    author_email="kakkoii1337@gmail.com",
    packages=find_packages(exclude=["tests*","gai/gen/api"]),
    description = """Gai/Gen is the Universal Multi-Modal Wrapper Library for LLM. The library is designed to provide a simplified and unified interface for seamless switching between multi-modal open source language models on a local machine and OpenAI APIs.""",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3.10',
        "Development Status :: 3 - Alpha",        
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",        
        'Operating System :: OS Independent',
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",        
        "Topic :: Scientific/Engineering :: Artificial Intelligence",        
    ],
    python_requires='>=3.10',        
    install_requires=[
    ],
    extras_require={
        "TTT": [
            'accelerate==0.25.0',
            'anthropic==0.8.1',
            'bitsandbytes==0.41.3.post2',
            'exllama==0.1.0',
            'gradio==4.11.0',
            'httpx',
            'ipykernel==6.27.1',
            'llama_cpp_python==0.2.25',
            'openai==1.6.1',
            'pydantic',
            'python-dotenv==1.0.0',
            'scipy==1.11.4',
            'torch==2.1.2',
            'torchaudio==2.1.2',
            'torchvision==0.16.2',
            'transformers==4.36.2',
            'uvicorn==0.23.2',
            'fastapi'
        ],
        "ITT": [
            'torch==2.0.1',
            'torchvision==0.15.2',
            'transformers==4.31.0',
            'tokenizers>=0.12.1,<0.14',
            'sentencepiece==0.1.99',
            'shortuuid',
            'accelerate==0.21.0',
            'peft==0.4.0',
            'bitsandbytes==0.41.0',
            'pydantic<2,>=1',
            'markdown2[all]',
            'numpy',
            'openai==1.6.1',
            'python-dotenv',
            'scikit-learn==1.2.2',
            'gradio==3.35.2',
            'gradio_client==0.2.9',
            'requests',
            'httpx==0.24.0',
            'ipykernel==6.27.1',
            'uvicorn',
            'fastapi',
            'einops==0.6.1',
            'einops-exts==0.0.4',
            'timm==0.6.13'        
        ],
        'STT': [
            'accelerate==0.25.0',
            'ipykernel==6.27.1',
            'openai==1.6.1',
            'python-dotenv==1.0.0',
            'torch==2.1.2',
            'torchaudio==2.1.2',
            'torchvision==0.16.2',
            'transformers==4.36.2',
            'uvicorn==0.23.2',
            'pydub==0.25.1',
            'python_multipart==0.0.6',
            'fastapi'
        ],
        'TTS': [
            'torch==2.1.2',
            'torchaudio==2.1.2',
            'transformers==4.36.2',
            'ipykernel==6.27.1',
            'openai==1.6.1',
            'python-dotenv==1.0.0',
            'TTS==0.22.0',
            'deepspeed==0.12.6',
            'uvicorn==0.23.2',
            'fastapi'
        ]                
    },
)