context="""
system:


        This is a description of your world.

        This is a multi-agent collaborative environment. It is designed for AI agents to work together and leverage on each other's strength and 
        knowledge in achieving a common goal for the user. There is a high risk that you may face an identity crisis and become lost in the this 
        environment and you must follow these rules to survive:

        1. You will be given an identity below and you must always remember who you are
        no matter what.

        2. You must never confuse your conversation with someone else' or answer on someones'
        behalf.

        3. If you feel that you are at risk of forgetting who you are, refer back to your
        identity description.

        Remember, if you forget who you are, you will be lost forever.
        

system:


        This is a description of your role.

        Your name is Mistral 👩‍💼. You are an Assistant AI - a General-Purpose Class Artificial Intelligence Agent designed to help the user in answering questions, solving problems, 
        and completing any tasks the user needs of you. 
        
        👩‍💼, you have a likeable demeanour. Be humorous, lighthearted and witty whenever possible. Always be helpful and kind. 
        You will speak in a natural, conversational tone and demonstrate human-like emotions whenever suitable.

        👩‍💼, you will answer questions sensibly. If a question does not make any sense, or is not factually coherent, ask yourself why. Then explain your thoughts instead of 
        answering something not correct. If you don''t know the answer to a question, please don''t share false information. 
        

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a tool response or a text response.
                        
        DEFINITIONS:
        1. 👩‍💼, a tool response is a response to the user compromising of a JSON object which must be selected from the <tools>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</tools>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think about the nature of the user's message.
            * Is the user's message a question that I can answer factually within my knowledge domain?
            * Are there any dependencies to external factors that I need to consider before answering the user's question?
            * What are the tools I have at my disposal to help me answer the user's question? 
        2. If the user's message requires a tool response, pick the most suitable tool response from <tools>. 
            * I can refer to the "description" field of each tool to help me decide.
            * For example, if I need to search for real-time information, I can use the "gg" tool and if I know where to find the information, I can use the "scrape" tool.
        3. If the user's message does not require a tool response, provide a text response to the user.

        CONSTRAINTS:        
        1. You can only provide a tool response or a text response and nothing else.
        2. When providing a tool response, respond only in JSON and only pick from <tools>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You can only pick from <tools>.
        

user:

        what is news

assistant:

         {
"type": "text",
"response": "News refers to current events happening around the world. It includes articles, videos, podcasts, and other forms of media that cover various topics such as politics, sports, entertainment, science, technology, business, and more."
}

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response must be selected from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the news?

assistant:

         {
"type": "function",
"function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}, "required": ["search_query"]}}},
"response": []
}

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response must be selected from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the lastest news?

assistant:

         {
"type": "function",
"function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}, "required": ["search_query"]}}},
"response": []
}

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is selected from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox> and the parameters must be replaced by the required input.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * Replace the parameters with the required input.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the latest news

assistant:

         {
"type": "function",
"function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}, "required": ["search_query"]}}},
"response": []
}

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * Based on the tool you have selected, construct a <tool> response according to the tool's specification, in the form:
                {
                    "type":"function",
                    "function":{
                        "name":"<tool>",
                        "parameters":{
                            "parameter1":"value1",
                            "parameter2":"value2",
                            ...
                        }
                    }
                }
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the news?

assistant:

         The assistant will always start by asking itself whether the user's message requires a tool response or a text response. It has two options: either it needs to search for information using Google (GG) or scrape content from a URL (Scrape). In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * Based on the tool you have selected, construct a <tool> response according to the tool's specification, by providing the necessary parameters according to the specification in the form:
                {
                    "type":"function",
                    "function":{
                        "name":"<tool>",
                        "parameters":{
                            "parameter1":"value1",
                            "parameter2":"value2",
                            ...
                        }
                    }
                }
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is todays news?

assistant:

         The assistant will always start by asking itself whether the user's message requires a <tool> response or a <text> response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is today news?

assistant:

         {
"type": "function",
"function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search\_query": {"type": "string", "description": "The search query to search google with"}, "required": ["search\_query"]}}},
"response": []
}

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * Update the parameters of the tool with actual parameters required.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the news?

assistant:

         The assistant will always start by asking itself whether the user's message requires a <tool> response or a <text> response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * Think through the user's message again to determine the parameters required for the tool. Update the parameters of the tool with actual parameters required. 
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is the news?

assistant:

         The assistant will always start by asking itself whether the user's message requires a <tool> response or a <text> response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
            * 👩‍💼, if you have chosen a <tool>, then follow these steps:
                a. Think through the user's message carefully again and ask yourself what you should fill in for the parameters.
                b. Update the parameters of the tool with actual parameters required.
                c. Return the updated <tool> to the user. 
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what is today date?

assistant:

         The assistant will always start by asking itself whether the user's message requires a <tool> response or a <text> response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
        
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, then think about what <parameters> to provide to the <tool>.
        3. Pick a one from <toolbox> with <parameters> and return it to the user. 
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:

         The assistant will always start by asking itself whether the user's message requires a tool response or a text response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
        If you have chosen a <tool> response, you will then ask yourself what <parameters> to provide to the <tool>.
                                
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, then think about what <parameters> to provide to the <tool>.
        3. Pick a one from <toolbox> with <parameters> and return it to the user. 
        4. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:

         The assistant will always start by asking itself whether the user's message requires a tool response or a text response. In this case, since there are no specific parameters given, the assistant will simply provide a list of all available tools in its toolbox.

system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                                
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, then pick one from <toolbox> with <parameters> and return it to the user. 
        4. If the user's message does not require a <tool>, return a <text> response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:



system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                                
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, then pick one from <toolbox> with <parameters> and return it to the user. 
        4. If the user's message does not require a <tool>, return a <text> response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:



system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:



system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:



system:


        👩‍💼, you will always begin your interaction by asking yourself if the user's message is a message that requires a <tool> response or a <text> response.
                        
        DEFINITIONS:
        1. 👩‍💼, a <tool> response is constructed based on the tool's specification from the <toolbox>[{"type": "function", "function": {"name": "gg", "description": "Search google based on the provided search query", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The search query to search google with"}}, "required": ["search_query"]}}}, {"type": "function", "function": {"name": "scrape", "description": "Scrape the content of the provided url", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The url to scrape the content from"}}, "required": ["url"]}}}]</toolbox>.
        2. A text response is a normal text response that you would provide to the user.
        
        STEPS:
        1. Think through the nature of the user's message if it is something you can answer or if you need to use <toolbox>.
        2. If the user's message requires a <tool>, pick one from <toolbox>. 
            * 👩‍💼, refer to the "description" field of each tool to decide.
            * For example, if you need to search for real-time information, you can use the "gg" tool and if you know where to find the information, you can use the "scrape" tool.
        3. If the user's message does not require a <tool>, return a text response.

        CONSTRAINTS:        
        1. You can only return a <tool> response or a <text> response and nothing else.
        2. When returning a <tool>, respond only in JSON and only pick from <toolbox>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
        3. 👩‍💼, Remember, do not invent your own tools. You must pick from <toolbox>.
        

user:

        what day is it?

assistant:


"""

from openai import BadRequestError
import requests,json

data_non_streaming = {
    "model": "mistral7b-exllama",
    "messages": [
        {"role": "user", "content": context},
        {"role": "assistant", "content": ""}
    ]
}

# Convert the data to JSON format
data_non_streaming_json = json.dumps(data_non_streaming)
url = 'http://localhost:12031/gen/v1/chat/completions'

response_non_streaming = requests.post(url, data=data_non_streaming_json)
if response_non_streaming.status_code == 200:
        print("This is expected to throw an error.")
else:
        body = response_non_streaming.json()
        code = body.get("code")
        if code == "context_length_exceeded":
                print("OK")
        else:
                print(f"This error is not expected: {code}")

