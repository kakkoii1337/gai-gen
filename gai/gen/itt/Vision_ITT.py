class Vision_ITT:
    def __init__(self, model_config):
        self.client = None
        pass

    def load(self):
        from dotenv import load_dotenv        
        load_dotenv()
        import os
        class MissingOpenAIApiKeyException(Exception):
            pass
        if 'OPENAI_API_KEY' not in os.environ:
            msg = "OPENAI_API_KEY not found in environment variables. GPT-4 will not be available."
            raise MissingOpenAIApiKeyException(msg)
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
        from openai import OpenAI
        self.client = OpenAI()

    def unload(self):
        self.client = None

    def gen(self, text, image_file, **model_params):
        import base64
        encoded_string = ""
        with open("./tests/buses.jpeg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_string}"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": image_url,
                        },
                    },
                ],
            }
        ]
        return self.create(messages, **model_params)

    def create(self, messages, **model_params):
        if not self.client:
            self.load()
        model_params.pop("model",None)
        stream = model_params.pop("stream",False)

        if stream:
            response = (chunk for chunk in self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                stream=True,
                **model_params
                ))
        else:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                **model_params
                )
        return response