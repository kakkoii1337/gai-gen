curl -X POST http://localhost:12031/gen/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"openai-tts-1\",\"input\":\"I think there is no direct bus. You can take 185 and change to MRT at buona vista. 185 should be arriving in 5 minutes.\"}" | ffplay -autoexit -nodisp -hide_banner -
