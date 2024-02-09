curl -X POST 'http://localhost:12031/gen/v1/rag/index-file' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -s \
    -F 'collection_name=demo' \
    -F 'file=@./attention-is-all-you-need.pdf' \
    -F 'metadata={"source": "https://arxiv.org/abs/1706.03762"}'