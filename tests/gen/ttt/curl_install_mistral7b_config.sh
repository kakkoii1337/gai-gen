curl -X PUT \
    http://localhost:12031/gen/v1/chat/config \
    -H 'Content-Type: application/json' \
    -s \
    -d '{"generator_name":"mistral7b-exllama",                       
        "generator_config": {                               
            "type": "ttt",                                  
            "model_name": "Mistral7B-ExLlama",              
            "engine": "ExLlama_TTT",                        
            "model_path": "models/Mistral-7B-Instruct-v0.1-GPTQ",   
            "model_basename": "model",                      
            "max_seq_len": 8192,                            
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
