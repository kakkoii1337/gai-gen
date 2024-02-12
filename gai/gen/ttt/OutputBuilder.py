from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from datetime import datetime
from uuid import uuid4

class OutputBuilder:

    @staticmethod
    def Generate_ChatCompletion_Id():
        return "chatcmpl-"+str(uuid4())

    @staticmethod
    def Generate_ToolCall_Id():
        return "call_"+str(uuid4())

    @staticmethod
    def Generate_CreationTime():
        return int(datetime.now().timestamp())

    @staticmethod
    def BuildTool(generator,function_name,function_arguments,prompt_tokens,new_tokens):
        return OutputBuilder(
            ).add_chat_completion(generator=generator
                ).add_choice(finish_reason='tool_calls'
                    ).add_tool(
                        function_name=function_name,
                        function_arguments=function_arguments
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()

    @staticmethod
    def BuildContent(generator,finish_reason, content,prompt_tokens,new_tokens):
        return OutputBuilder(
            ).add_chat_completion(generator=generator
                ).add_choice(finish_reason=finish_reason
                    ).add_content(
                        content=content
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()

    def add_chat_completion(self,generator):
        chatcompletion_id = OutputBuilder.Generate_ChatCompletion_Id()
        created = OutputBuilder.Generate_CreationTime()
        self.result = ChatCompletion(
            id=chatcompletion_id,
            choices=[],
            created=created,
            model=generator,
            object='chat.completion',
            usage=None
        )
        return self

    def add_choice(self,finish_reason):
        self.result.choices.append(Choice(
            finish_reason=finish_reason,
            index=0,
            message=ChatCompletionMessage(role='assistant',content=None, function_call=None, tool_calls=[])
        ))
        return self

    def add_tool(self,function_name,function_arguments):
        toolcall_id = OutputBuilder.Generate_ToolCall_Id()
        self.result.choices[0].message.tool_calls.append(ChatCompletionMessageToolCall(
            id = toolcall_id,
            function = Function(
                name=function_name,
                arguments=function_arguments
            ),
            type='function'
        ))
        return self

    def add_content(self,content):
        self.result.choices[0].message.content = content
        self.result.choices[0].message.tool_calls = None
        return self
    
    def add_usage(self, prompt_tokens, new_tokens):
        total_tokens = prompt_tokens + new_tokens
        self.result.usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=new_tokens,
            total_tokens=total_tokens
        )
        return self
    
    def build(self):
        return self.result.copy()
