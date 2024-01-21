from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from datetime import datetime
from uuid import uuid4

class OutputBuilder:

    @staticmethod
    def Generate_ChatCompletion_Id():
        return "chatcmpl-"+str(uuid4())

    def Generate_ToolCall_Id():
        return "call-"+str(uuid4())

    def Generate_CreationTime():
        return int(datetime.now().timestamp())

    def Build_Tool_Start(model,created,chunk_id,role,tool_id,tool_name):
        builder = OutputBuilder(
        ).add_chunk(model=model,created=created,id=chunk_id
            ).add_chunk_choice_delta(finish_reason=None, role=role
                ).add_chunk_choice_delta_function(id=tool_id,name=tool_name)
        return builder.build()

    def Build_Tool_Body(model,created,chunk_id,tool_arg):
        builder = OutputBuilder(
            ).add_chunk(model=model,created=created,id=chunk_id
                ).add_chunk_choice_delta(finish_reason=None, role=None
                    ).add_chunk_choice_delta_function(arg=tool_arg,type=None)
        return builder.build()

    def Build_Tool_End(model,created,chunk_id,finish_reason):
        builder = OutputBuilder(
            ).add_chunk(model=model,created=created,id=chunk_id
                ).add_chunk_choice_delta(finish_reason=finish_reason, role=None)
        return builder.build()

    def __init__(self, result=None):
        self.result = None
        if result:
            self.result = result.copy()

    def copy(self):
        return OutputBuilder(self.result)

    def add_chunk(self,model,id=None,created=None):
        if not created:
            created = int(datetime.now().timestamp())
        if id=='':
            id = str(uuid4())
        self.result = ChatCompletionChunk(
            id=id,
            choices=[],
            created=created,
            model=model,
            object='chat.completion.chunk',
            system_fingerprint=None
        )
        return self

    def add_chunk_choice_delta(self,finish_reason=None, role: str = "assistant"):
        delta = ChoiceDelta(
            content=None, 
            role=role, 
            tool_calls=None, 
            function_call=None
            )
        choice = ChunkChoice(
            delta=delta,
            index=0,
            logprobs=None,
            finish_reason=finish_reason,
        )
        self.result.choices.append(choice)
        return self
    
    def add_chunk_choice_delta_function(self, id=None, name=None, arg='', index=0, type='function'):
        if name==None and arg=='':
            raise Exception("Name and Argument cannot both be empty")
        if id=='':
            id = 'call_'+str(uuid4())
        tool = ChoiceDeltaToolCall(
            index=index,             
            id=id, 
            function=ChoiceDeltaToolCallFunction(
                name=name, 
                arguments=arg
                ),
            type=type)

        if not self.result.choices[0].delta.tool_calls:
            self.result.choices[0].delta.tool_calls = [tool]
        else:
            self.result.choices[0].delta.tool_calls.append(tool)
        return self
    
    def build(self):
        return self.result.copy()

