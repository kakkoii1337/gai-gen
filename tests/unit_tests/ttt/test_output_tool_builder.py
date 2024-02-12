from gai.gen.ttt.OutputBuilder import OutputBuilder
from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder
import unittest
import json



class TestOutputBuilder(unittest.TestCase):

    # Target: ChatCompletion(id='chatcmpl-', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_', function=Function(arguments='{\n  "search_query": "latest news on Singapore"\n}', name='gg'), type='function')]))], created=1707656866, model='gpt-4-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=101, total_tokens=119)
    # Output: ChatCompletion(id='chatcmpl-', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_', function=Function(arguments="{'search_query':'latest news on Singapore'}", name='gg'), type='function')]))], created=1707659495, model='mistral7b-exllama', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=101, total_tokens=119))
    def test_build_tool(self):
        result = OutputBuilder.BuildTool(
            generator="mistral7b-exllama",
            function_name="gg", 
            function_arguments='{"search_query":"latest news on Singapore"}',
            prompt_tokens=101,
            new_tokens=18)
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion')
        self.assertEqual(result.choices[0].finish_reason, 'tool_calls')
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].message.role, 'assistant')
        self.assertEqual(result.choices[0].message.tool_calls[0].function.name, 'gg')
        self.assertEqual(json.loads(result.choices[0].message.tool_calls[0].function.arguments), {'search_query':'latest news on Singapore'})
        self.assertEqual(result.usage.prompt_tokens, 101)
        self.assertEqual(result.usage.completion_tokens, 18)

    # Target 1: ChatCompletion(id='chatcmpl-', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content="In the quiet town of Meadowgrove, lived a humble baker named Tom. Known for his enchanting blueberry pies, Tom's warm-hearted nature and magical baking skills had made him beloved by all. One cold winter's night, a mysterious stranger arrived, claiming he could not taste anything due to a curse. Hearing this, Tom worked through the night, using his secret ingredient, a pinch of love. He presented the pie to the stranger, who took a bite and his eyes lit up as", role='assistant', function_call=None, tool_calls=None))], created=1707663640, model='gpt-4-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=100, prompt_tokens=25, total_tokens=125))
    # Output:   ChatCompletion(id='chatcmpl-', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content="In the quiet town of Meadowgrove, lived a humble baker named Tom. Known for his enchanting blueberry pies, Tom's warm-hearted nature and magical baking skills had made him beloved by all. One cold winter's night, a mysterious stranger arrived, claiming he could not taste anything due to a curse. Hearing this, Tom worked through the night, using his secret ingredient, a pinch of love. He presented the pie to the stranger, who took a bite and his eyes lit up as", role='assistant', function_call=None, tool_calls=None))], created=1707664662, model='mistral7b-exllama', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=100, prompt_tokens=25, total_tokens=125))
    def test_build_content(self):
        result = OutputBuilder.BuildContent(
            generator="mistral7b-exllama",
            finish_reason="length",
            content="In the quiet town of Meadowgrove, lived a humble baker named Tom. Known for his enchanting blueberry pies, Tom's warm-hearted nature and magical baking skills had made him beloved by all. One cold winter's night, a mysterious stranger arrived, claiming he could not taste anything due to a curse. Hearing this, Tom worked through the night, using his secret ingredient, a pinch of love. He presented the pie to the stranger, who took a bite and his eyes lit up as",
            prompt_tokens=25,
            new_tokens=100)
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion')
        self.assertEqual(result.choices[0].finish_reason, 'length')
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].message.role, 'assistant')
        self.assertEqual(result.choices[0].message.tool_calls, None)        
        self.assertEqual(result.usage.prompt_tokens, 25)
        self.assertEqual(result.usage.completion_tokens, 100)
    
    # target head: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[ChoiceDeltaToolCall(index=0, id='call_', function=ChoiceDeltaToolCallFunction(arguments='', name='gg'), type='function')]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[ChoiceDeltaToolCall(index=0, id='call_', function=ChoiceDeltaToolCallFunction(arguments='', name='gg'), type='function')]), finish_reason=None, index=0, logprobs=None)], created=1707708338, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    def test_chunk_tool_head(self):
        result = ChunkOutputBuilder.BuildToolHead(generator="mistral7b-exllama",tool_name="gg")
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, None)
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, 'assistant')
        self.assertEqual(result.choices[0].delta.tool_calls[0].function.name, 'gg')        
        self.assertEqual(result.choices[0].delta.tool_calls[0].function.arguments, '')        
        self.assertEqual(result.choices[0].delta.content, None)

    # target body: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{\n', name=None), type=None)]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{{\n', name=None), type=None)]), finish_reason=None, index=0, logprobs=None)], created=1707708958, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    def test_chunk_tool_body(self):
        result = ChunkOutputBuilder.BuildToolBody(generator="mistral7b-exllama",tool_arguments="{\n")
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, None)
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, None)
        self.assertEqual(result.choices[0].delta.tool_calls[0].function.name, None)        
        self.assertEqual(result.choices[0].delta.tool_calls[0].function.arguments, '{\n')        
        self.assertEqual(result.choices[0].delta.content, None)

    # target tail: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='tool_calls', index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='tool_calls', index=0, logprobs=None)], created=1707709360, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)   
    def test_chunk_tool_tail(self):
        result = ChunkOutputBuilder.BuildToolTail(generator="mistral7b-exllama",finish_reason='tool_calls')
        print(result)
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, 'tool_calls')
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, None)
        self.assertEqual(result.choices[0].delta.tool_calls, None)        
        self.assertEqual(result.choices[0].delta.content, None)

    # target head: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707705640, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    def test_chunk_content_head(self):
        result = ChunkOutputBuilder.BuildContentHead(generator="mistral7b-exllama")
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, None)
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, 'assistant')
        self.assertEqual(result.choices[0].delta.tool_calls, None)        
        self.assertEqual(result.choices[0].delta.content, '')

    # target body: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content='Once', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content='Once', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707706176, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    def test_chunk_content_body(self):
        result = ChunkOutputBuilder.BuildContentBody(generator="mistral7b-exllama",content='Once')
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, None)
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, None)
        self.assertEqual(result.choices[0].delta.tool_calls, None)        
        self.assertEqual(result.choices[0].delta.content, 'Once')
        
    # target tail: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='length', index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    # output:      ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='length', index=0, logprobs=None)], created=1707706371, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    def test_chunk_content_tail(self):
        result = ChunkOutputBuilder.BuildContentTail(generator="mistral7b-exllama",finish_reason='length')
        self.assertEqual(result.model, "mistral7b-exllama")
        self.assertEqual(result.object, 'chat.completion.chunk')
        self.assertEqual(result.choices[0].finish_reason, 'length')
        self.assertEqual(result.choices[0].index, 0)
        self.assertEqual(result.choices[0].delta.role, None)
        self.assertEqual(result.choices[0].delta.tool_calls, None)        
        self.assertEqual(result.choices[0].delta.content, None)
        

if __name__ == '__main__':
    unittest.main()