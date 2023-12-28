# Gai/Gen: Universal LLM Wrapper

## 1. Introduction

This is a Universal Multi-Modal Wrapper Library for LLM inferencing.

The library provides a simplified and unified interface for seamless switching between multi-modal open source language models on a local machine and OpenAI APIs.

This is intended for Developers who are targetting the use of multi-modal LLMs for both OpenAI API and local machine models.

The core object is called **Gaigen** - generative AI generator. The premise is for the code to run on as commonly available commodity hardware as possible. The main focus is on 7 billion parameters and below open source models. Gaigen is designed as singleton wrapper where only one model is loaded and cached into memory at any one time.
To avoid dependency conflicts, the wrappers are organised under the `gen` folder according to 4 mutually-exclusive categories:

-   ttt: Text-to-Text
-   tts: Text-to-Speech
-   stt: Speech-to-Text
-   itt: Image-to-Text

## 2. Requirements

-   The instructions are tested mainly on:
    -   Windows 11 with WSL
    -   Ubuntu 20.04.2 LTS
    -   NVIDIA RTX 2060 GPU with 8GB VRAM
    -   CUDA Toolkit is required for the GPU accelerated models. Run `nvidia-smi` to check if CUDA is installed.
        If you need help, refer to this https://gist.github.com/kakkoii1337/8a8d4d0bc71fa9c099a683d1601f219e

## 3. Credits

This library is made possible by the generosity and hardwork of the following open source projects. You are highly encouraged to check out the original source and documentations.

TTT

-   [TheBloke](https://huggingface.co/TheBloke) for all the quantized models in the demo
-   [turboderp](https://github.com/turboderp/exllama) for ExLlama
-   Meta Team for the [LLaMa2](https://ai.meta.com/llama/) Model
-   HuggingFace team for the [Transformers](https://huggingface.co/docs/transformers/llm_tutorial) library and open source models
-   [Mistral AI Team for [Mistral7B](https://mistral.ai/news/announcing-mistral-7b/) Model
-   Georgi Gerganov for [LLaMaCpp](https://github.com/ggerganov/llama.cpp)

ITT

-   Liu HaoTian for the [LLaVa](https://github.com/haotian-liu/LLaVA) Model and Library

TTS

-   [Coqui-AI](https://github.com/coqui-ai/TTS) for the xTTS Model

STT

-   [OpenAI](https://huggingface.co/openai/whisper-large-v3) for Open Sourcing Whisper v3

---

## 4. Using Gai as a Library

Using Gai as a library requires you to install the right category of package but gives you more control over your interaction with Gaigen.
It is highly recommended that you install each category in separate virtual environments.

```bash
# Install library for text-to-text generation
pip install gai-lib-gen[TTT]

# Install library for text-to-speech generation
pip install gai-lib-gen[TTS]

# Install library for speech-to-text generation
pip install gai-lib-gen[STT]

# Install library for image-to-text generation
pip install gai-lib-gen[ITT]
```

### Configuration

Step 1. Create a `.gairc` file in your home directory. This file contains the default configurations for Gai.

```bash
{
    "app_dir": "~/gai"
}
```

Step 2. Create a `/gai` directory.

```bash
mkdir ~/gai
```

Copy `gai.json` from this repository into `~/gai`. This file contains the configurations for models and their respective loaders.

Step 3. Create `/gai/models` directory.

```bash
mkdir ~/gai/models
```

The final user directory structure looks like this:

```bash
home
├── gai
│   ├── gai.json
│   └── models
└── .gairc
```

### API Key

-   All API keys should be stored in a `.env` file in the root directory of the project. For example,

    ```.env
    OPENAI_API_KEY=<--replace-with-your-api-key-->
    ANTHROPIC_API_KEY=<--replace-with-your-api-key-->
    ```

### Quick Start

**1. Install virtal environment and Gai**

```bash
conda create -n TTT python=3.10.10 -y
conda activate TTT
pip install gai-lib-gen[TTT]
```

**1. Setup OpenAI API Key.**

Save your OpenAI API key in the **.env** file in the root directory of your project.

```bash
OPENAI_API_KEY=<--replace-with-your-api-key-->
```

**2. Run GPT4.**

Run Text-to-Text generation using OpenAI by loading `gpt-4` wrapper.

```python
from gai.gen import Gaigen
gen = Gaigen.GetInstance().load('gpt-4')

response = gen.create(messages=[{'role':'USER','content':'Tell me a one paragraph short story.'},{'role':'ASSISTANT','content':''}])
print(response)
```

**3. Install Mistral7B.**

Open `~/gai/gai.json` and refer to `model_path`:

-   under `mistral7b-exllama`, refer to `model-path` for the repo name.
-   Go to huggingface
-   Download the repo `Mistral-7B-Instruct-v0.1-GPTQ` into the `~/gai/models` folder.

**4. Run Mistral7B.**

Run Text-to-Text generation using Mistral7B by replacing `gpt-4` with `mistral7b-exllama`.

```python
from gai.gen import Gaigen
gen = Gaigen.GetInstance().load('mistral7b-exllama')

response = gen.create(messages=[{'role':'USER','content':'Tell me a one paragraph short story.'},{'role':'ASSISTANT','content':''}])
print(response)
```

### Examples

-   [Text-to-Text Generation](/docs/TTT.ipynb)
-   [Speech-to-Text Generation](/docs/STT.ipynb)
-   [Text-to-Speech Generation](/docs/TTS.ipynb)
-   [Image-to-Text Generation](/docs/ITT.ipynb)

## 5. Using Gai as a Service

The simplest way to use Gai is to run it as a service using Docker.
You will still need to download the models into ~/gai/models but map the volume to the container.
You can then start up a container and post REST API calls to a single endpoint:

### Endpoints

The endpoints corresponding to each categories are:

**- Text-to-Text (TTT)**  
Endpoint: http://localhost:12031/gen/v1/chat/completions  
Method: POST  
Body:

```json
{
    "model": "More specifically, this is the name of the generator, eg. "mistral7b-exllama" or "gpt-4" etc, refer to the gai.json keys as a reference,
    "messages": Follows openai styled message list, [{"role":user|system|ai, "content":message}],
    "stream": true|false,
    hyperparameters(eg. these will correspond to the parameters based on the model parameter specified above)
}
```

**- Text-to-Speech (TTS)**  
Endpoint: http://localhost:12031/gen/v1/audio/speech  
Method: POST  
Body:

```json
{
    "model": "More specifically, this is the name of the generator, eg. "xtts-2" or "openai-tts" etc, refer to the gai.json keys as a reference,
    "input": text to be spoken,
    "voice": speaker name (this will correspond to the list provided by the model parameter specified above),
    "language": language code (this will correspond to the list provided by the model parameter specified above),
    "stream": true|false,
    hyperparameters(eg. these will correspond to the parameters based on the model parameter specified above)
}
```

**- Speech-to-Text**
Endpoint: http://localhost:12031/gen/v1/audio/transcriptions  
Method: POST  
www-encoded-form:
model: string
file: file

**- itt: Image-to-Text**
Endpoint: http://localhost:12031/gen/v1/vision/completions  
Method: POST

```json
Body: {
    "model": "More specifically, this is the name of the generator, eg. "llava-transformers" or "openai-vision" etc, refer to the gai.json keys as a reference,
    "messages": Follows openai styled message list, [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": 'data:image/jpeg;base64,.....',
                    },
                },
            ],
            ...
        }
    ],
    "stream": true|false,
    hyperparameters(eg. these will correspond to the parameters based on the model parameter specified above)
}
```

You can start the container for the corresponding LLM category and start using via REST API calls.

### Examples

-   [Text-to-Text Generation](/docs/TTT.ipynb)
-   [Speech-to-Text Generation](/docs/STT.ipynb)
-   [Text-to-Speech Generation](/docs/TTS.ipynb)
-   [Image-to-Text Generation](/docs/ITT.ipynb)
