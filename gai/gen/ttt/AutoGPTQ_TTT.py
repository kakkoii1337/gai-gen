from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch, gc, re
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

class AutoGPTQ_TTT:

    param_whitelist=[
        'do_sample',
        'early_stopping',
        'encoder_repetition_penalty',
        'eos_token_id',
        'length_penalty',
        'logits_processor',
        'max_new_tokens',
        'min_length',
        'no_repeat_ngram_size',
        'num_beams',
        'penalty_alpha',
        'repetition_penalty',
        'stopping_criteria',
        'temperature',
        'top_k',
        'top_p',
        'typical_p'
        ]

    def get_model_params(self, **kwargs):
        params = {
            'max_new_tokens': 25,
            'do_sample': True,
            'early_stopping': False,
            'encoder_repetition_penalty': 1,
            'eos_token_id': self.tokenizer.eos_token_id,
            'length_penalty': 1,
            'logits_processor': [],
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'repetition_penalty': 1.17,
            'temperature': 1.31,
            'top_k': 49,
            'top_p': 0.14,
            'typical_p': 1
        }
        return {**params,**kwargs}

    def __init__(self,model_config):
        if (model_config is None):
            raise Exception("autogptq_engine: model_config is required")
        if "model_path" not in model_config or model_config["model_path"] is None:
            raise Exception("autogptq_engine: model_path is required")
        if "model_basename" not in model_config or model_config["model_basename"] is None:
            raise Exception("autogptq_engine: model_basename is required")

        self.config = model_config
        self.model_filepath = f'{model_config["model_path"]}/{model_config["model_basename"]}.safetensors'
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load(self):
        logger.info(f"Loading model from {self.model_filepath}")
        use_triton = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], use_fast=True, legacy=False)
        self.model = AutoGPTQForCausalLM.from_quantized(self.config["model_path"],
            model_basename=self.config["model_basename"],
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None)
        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.generator
        except :
            pass
        self.model = None
        self.tokenizer = None
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self,text):
        return len(self.tokenizer.tokenize(text))

    def get_response(self,output,ai_role="ASSISTANT"):
        return re.split(rf'{ai_role}:', output, flags=re.IGNORECASE)[-1].strip().replace('\n\n', '\n').replace('</s>', '')

    def generate(self,prompt,ai_role="ASSISTANT",**model_params):
        logger.debug(f"generate: prompt={prompt}")

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        logger.debug(f"model_params: {model_params}")

        input_ids = self.tokenizer(prompt,return_tensors="pt",add_special_tokens=False).input_ids.cuda()
        input_count=self.token_count(prompt)
        logger.debug(f"generate: input token count={input_count}")

        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200
        outputs = self.model.generate(inputs=input_ids, **model_params).cuda()
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"generate: raw output={output}")

        output_count = self.token_count(output)
        logger.debug(f"generate: output token count={str(output_count)}")

        logger.info(f"generate: output token count={str(output_count)}  < max_new_tokens: {str(max_new_tokens)}")
        return self.get_response(output,ai_role)
        
    def streaming(self,prompt,ai_role="ASSISTANT",**model_params):
        raise Exception("autogptq_engine: streaming not supported")