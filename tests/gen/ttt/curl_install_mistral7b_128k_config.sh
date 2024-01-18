curl -X PUT \
    http://localhost:12031/gen/v1/chat/config \
    -H 'Content-Type: application/json' \
    -s \
    -d '{"generator_name":"mistral7b_128k-exllama",                       
        "generator_config": {                               
            "type": "ttt",
            "model_name": "Mistral7B_128k-ExLlama",
            "engine": "ExLlamaV2_TTT",
            "model_path": "models/Yarn-Mistral-7B-128k-GPTQ",
            "model_basename": "model",
            "max_seq_len": 65536,
            "stopping_words": [],
            "hyperparameters": {
                "temperature": 1.2,
                "top_p": 0.15,
                "min_p": 0.0,
                "top_k": 50,
                "max_new_tokens": 1000,
                "typical": 0.0,
                "token_repetition_penalty_max": 1.25,
                "token_repetition_penalty_sustain": 256,
                "token_repetition_penalty_decay": 128,
                "beams": 1,
                "beam_length": 1
            }
        }
    }'
