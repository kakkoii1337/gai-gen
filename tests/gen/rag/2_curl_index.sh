curl -X POST 'http://localhost:12031/gen/v1/rag/index-file' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -s \
    -F 'collection_name=demo' \
    -F 'file=@./pm_long_speech_2023.txt' \
    -F 'metadata={"source": "https://www.pmo.gov.sg/Newsroom/National-Day-Rally-2023#:~:text=COVID%2D19%20was%20the%20most,indomitable%20spirit%20of%20our%20nation."}'