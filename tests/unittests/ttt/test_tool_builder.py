from gai.gen.ttt.OutputBuilder import OutputBuilder
import json

## Head
model = "mistral7b-exllama"
created = 1705840897
chunk_id = "chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U"
role="assistant"
tool_id = "call_dDVySGhkam2r62PG1R4SqW1h"
tool_name="gg"
builder = OutputBuilder(
    ).add_chunk(model=model,created=created,id=chunk_id
        ).add_chunk_choice_delta(finish_reason=None, role=role
            ).add_chunk_choice_delta_function(id=tool_id,name=tool_name)
result = str(builder.build())
target = "ChatCompletionChunk(id='chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[ChoiceDeltaToolCall(index=0, id='call_dDVySGhkam2r62PG1R4SqW1h', function=ChoiceDeltaToolCallFunction(arguments='', name='gg'), type='function')]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)"
assert result == target, f"result: {result}\ntarget: {target}"

## Body
model = "mistral7b-exllama"
created = 1705840897
chunk_id = "chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U"
tool_arg="{\n"
builder = OutputBuilder(
    ).add_chunk(model=model,created=created,id=chunk_id
        ).add_chunk_choice_delta(finish_reason=None, role=None
            ).add_chunk_choice_delta_function(arg=tool_arg,type=None)
result = str(builder.build())
target = "ChatCompletionChunk(id='chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{\n', name=None), type=None)]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)"
assert result == repr(target)[1:-1], f"result: {result}\ntarget: {target}"

## Tail
finish_reason='tool_calls'
builder = OutputBuilder(
    ).add_chunk(model=model,created=created,id=chunk_id
        ).add_chunk_choice_delta(finish_reason=finish_reason, role=None)
result = str(builder.build())
target = "ChatCompletionChunk(id='chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='tool_calls', index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)"
assert result == repr(target)[1:-1], f"result: {result}\ntarget: {target}"