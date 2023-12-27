curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"gpt-4\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"stream\":true,\
        \"max_tokens\":25\
        }" \
        | python print_ttt_delta.py