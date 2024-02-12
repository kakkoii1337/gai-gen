from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from datetime import datetime
from uuid import uuid4

class ChunkOutputBuilder:

    @staticmethod
    def Generate_ChatCompletion_Id():
        return "chatcmpl-"+str(uuid4())

    @staticmethod
    def Generate_ToolCall_Id():
        return "call_"+str(uuid4())

    @staticmethod
    def Generate_CreationTime():
        return int(datetime.now().timestamp())

    # target head: ChatCompletionChunk(id='chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role='assistant', tool_calls=[ChoiceDeltaToolCall(index=0, id='call_dDVySGhkam2r62PG1R4SqW1h', function=ChoiceDeltaToolCallFunction(arguments='', name='gg'), type='function')]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildToolHead(generator,tool_name):
        builder = ChunkOutputBuilder(
        ).add_chunk(generator=generator
            ).add_chunk_choice_delta(finish_reason=None, role='assistant'
                ).add_chunk_choice_delta_toolcall_name(name=tool_name)
        return builder.build()

    # target body: ChatCompletionChunk(id='chatcmpl-', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{\n', name=None), type=None)]), finish_reason=None, index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildToolBody(generator,tool_arguments):
        builder = ChunkOutputBuilder(
        ).add_chunk(generator=generator
            ).add_chunk_choice_delta(finish_reason=None, role=None
                ).add_chunk_choice_delta_toolcall_arguments(arguments=tool_arguments)
        return builder.build()

    # target tail: ChatCompletionChunk(id='chatcmpl-8jRQn0D7LfyZBIXzknzEv2zBNDA9U', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='tool_calls', index=0, logprobs=None)], created=1705840897, model='mistral7b-exllama', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildToolTail(generator,finish_reason):
        builder = ChunkOutputBuilder(
            ).add_chunk(generator=generator
                ).add_chunk_choice_delta(finish_reason=finish_reason, role=None)
        return builder.build()

    # target head: ChatCompletionChunk(id='chatcmpl-8rG8XsWivUZSfin42AoIWS8zhkx8o', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildContentHead(generator):
        builder = ChunkOutputBuilder(
            ).add_chunk(generator=generator
                ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                    ).add_chunk_choice_delta_content(content='')
        return builder.build()

    # ChatCompletionChunk(id='chatcmpl-8rG8XsWivUZSfin42AoIWS8zhkx8o', choices=[Choice(delta=ChoiceDelta(content='Once', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildContentBody(generator,content):
        builder = ChunkOutputBuilder(
            ).add_chunk(generator=generator
                ).add_chunk_choice_delta(finish_reason=None, role=None
                    ).add_chunk_choice_delta_content(content=content)
        return builder.build()

    # ChatCompletionChunk(id='chatcmpl-8rG8XsWivUZSfin42AoIWS8zhkx8o', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='length', index=0, logprobs=None)], created=1707704105, model='gpt-4-0613', object='chat.completion.chunk', system_fingerprint=None)
    @staticmethod
    def BuildContentTail(generator,finish_reason):
        builder = ChunkOutputBuilder(
            ).add_chunk(generator=generator
                ).add_chunk_choice_delta(finish_reason=finish_reason, role=None
                    )
        return builder.build()

    def __init__(self, result=None):
        self.result = None
        if result:
            self.result = result.copy()

    def copy(self):
        return ChunkOutputBuilder(self.result)

    def add_chunk(self,generator):
        chatcompletion_id = ChunkOutputBuilder.Generate_ChatCompletion_Id()
        created = ChunkOutputBuilder.Generate_CreationTime()
        self.result = ChatCompletionChunk(
            id=chatcompletion_id,
            choices=[],
            created=created,
            model=generator,
            object='chat.completion.chunk'
        )
        return self

    def add_chunk_choice_delta(self, role=None, finish_reason=None):
        self.result.choices.append(
            ChunkChoice(
                delta=ChoiceDelta(
                    content=None, 
                    role=role, 
                    tool_calls=None, 
                    function_call=None
                    ),
                index=0,
                logprobs=None,
                finish_reason=finish_reason,
            )            
        )
        return self

    def add_chunk_choice_delta_content(self, content):
        self.result.choices[0].delta.content=content
        return self

    def add_chunk_choice_delta_toolcall_name(self, name):
        self.result.choices[0].delta.tool_calls=[ChoiceDeltaToolCall(
            index=0,
            id=ChunkOutputBuilder.Generate_ToolCall_Id(),
            function=ChoiceDeltaToolCallFunction(
                name=name,
                arguments=''
                ),
            type='function'
            )]
        return self

    def add_chunk_choice_delta_toolcall_arguments(self, arguments):
        self.result.choices[0].delta.tool_calls=[ChoiceDeltaToolCall(
            index=0,
            id=None,
            function=ChoiceDeltaToolCallFunction(
                name=None,
                arguments=arguments
                ),
            type='function'
            )]
        return self
    
    # def add_chunk_choice_delta_function(self, id=None, name=None, arg='', index=0, type='function'):
    #     if name==None and arg=='':
    #         raise Exception("Name and Argument cannot both be empty")
    #     if id=='':
    #         id = 'call_'+str(uuid4())
    #     tool = ChoiceDeltaToolCall(
    #         index=index,             
    #         id=id, 
    #         function=ChoiceDeltaToolCallFunction(
    #             name=name, 
    #             arguments=arg
    #             ),
    #         type=type)

    #     if not self.result.choices[0].delta.tool_calls:
    #         self.result.choices[0].delta.tool_calls = [tool]
    #     else:
    #         self.result.choices[0].delta.tool_calls.append(tool)
    #     return self
    
    def build(self):
        return self.result.copy()

