def print_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            print("CONTENT")
            print(chunk.choices[0].delta.content)
        elif chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.name:
            print("NAME")
            print(chunk.choices[0].delta.tool_calls[0].function.name)
        elif chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.arguments:
            print("ARGUMENTS")
            print(chunk.choices[0].delta.tool_calls[0].function.arguments)
        elif chunk.choices[0].finish_reason:
            print("FINISH REASON")
            print(chunk.choices[0].finish_reason)
        else:
            print(chunk.choices[0])
