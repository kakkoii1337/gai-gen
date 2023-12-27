curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"mistral7b-exllama\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"max_new_tokens\":25,
        \"stream\":true}" \
        | python print_ttt_delta.py