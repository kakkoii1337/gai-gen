curl -X POST 'http://localhost:12031/gen/v1/rag/retrieve' \
    -H "Content-Type: application/json" \
    -d '{"collection_name":"demo","query_texts":"Who are the young seniors?","n_results":4}'
