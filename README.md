# Gai/Gen: Universal LLM Wrapper

This is a Universal Multi-Modal Wrapper Library for LLM inferencing.

The library provides a simplified and unified interface for seamless switching between multi-modal open source language models on a local machine and OpenAI APIs.

This is intended for Developers who are targetting the use of multi-modal LLMs for both OpenAI API and local machine models.

As the main focus is on 7 billion parameters and below open source models, running on commodity hardware, the library is designed to be a singleton wrapper. Only one model is loaded and cached into memory at any one time.

The wrappers are organised under the `gen` folder according to 4 categories:

-   ttt: Text-to-Text
-   tts: Text-to-Speech
-   stt: Speech-to-Text
-   itt: Image-to-Text

---

## Setting Up

It is highly recommended that you install the following variants in separate virtual environments.

```bash
# Install library for text-to-text generation
pip install gai-gen[TTT]

# Install library for text-to-speech generation
pip install gai-gen[TTS]

# Install library for speech-to-text generation
pip install gai-gen[STT]

# Install library for image-to-text generation
pip install gai-gen[ITT]
```

or

```bash
git clone https://www.github.com/kakkoii1337/gai-gen
```

## Configuration

-   Create the default application directory `~/gai` and the default models directory `~/gai/models`. If you want to change this defaults, change `~/.gairc`

-   Copy `gai.json` from this repository into `~/gai`. This file contains the configurations for models and their respective loaders.

## API Key

-   All API keys should be stored in a `.env` file in the root directory of the project. For example,

    ```.env
    OPENAI_API_KEY=<--replace-with-your-api-key-->
    ANTHROPIC_API_KEY=<--replace-with-your-api-key-->
    ```

## Requirements

-   The instructions are tested mainly on:
    -   Windows 11 with WSL
    -   Ubuntu 20.04.2 LTS
    -   NVIDIA RTX 2060 GPU with 8GB VRAM
    -   CUDA Toolkit is required for the GPU accelerated models. Run `nvidia-smi` to check if CUDA is installed.
        If you need help, refer to this https://gist.github.com/kakkoii1337/8a8d4d0bc71fa9c099a683d1601f219e

## Credits

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

## Quick Start

**1. Setup OpenAI API Key.**

Save your OpenAI API key in a .env file in the root directory of your project.

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

-   under `mistral-8k`, refer to `model-path` for the repo name.
-   Go to huggingface
-   Download the repo `Mistral-7B-Instruct-v0.1-GPTQ` into the `~/gai/models` folder.

**4. Run Mistral7B.**

Run Text-to-Text generation using Mistral7B by replacing `gpt-4` with `mistral-8k`.

```python
from gai.gen import Gaigen
gen = Gaigen.GetInstance().load('mistral-8k')

response = gen.create(messages=[{'role':'USER','content':'Tell me a one paragraph short story.'},{'role':'ASSISTANT','content':''}])
print(response)
```

## Examples

-   [Text-to-Text Generation](/docs/TTT.ipynb)
-   [Speech-to-Text Generation](/docs/STT.ipynb)
-   [Text-to-Speech Generation](/docs/TTS.ipynb)
-   [Image-to-Text Generation](/docs/ITT.ipynb)
