curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"claude2-100k\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"max_tokens_to_sample\":25,\
        \"stream\":true}" \
        | python print_ttt_delta.py        