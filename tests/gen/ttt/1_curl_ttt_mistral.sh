echo STREAMING
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
        \"stream\":true}" \
        | python ../print_delta.py
echo
echo NON-STREAMING
curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"mistral7b-exllama\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ]}"

