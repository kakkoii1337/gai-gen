curl -X 'POST' \
'http://localhost:12031/gen/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -s \
    -F 'file=@../today-is-a-wonderful-day.wav' \
    -F 'model=whisper-transformers'