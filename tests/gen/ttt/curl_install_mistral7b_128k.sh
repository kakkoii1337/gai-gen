curl -X PUT \
    http://localhost:12031/gen/v1/chat/install \
    -H 'Content-Type: application/json' \
    -s \
    -d '{"repo":"TheBloke", "model":"Yarn-Mistral-7B-128k-GPTQ"}'
