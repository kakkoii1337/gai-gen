curl -X POST http://localhost:12031/gen/v1/vision/completions \
    -H "Content-Type: application/json" \
    -N \
    -d "{\"model\":\"openai-vision\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\": \"text\", \"text\": \"What’s in this image?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,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\"}}]}]}" 