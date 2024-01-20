from gai.common.generators_utils import chat_list_to_INST
input=[{"role":"system","content":"This is system_messge_1"},
       {"role":"system","content":"This is system_messge_2"},
       {"role":"system","content":"This is system_messge_3"},
       {"role":"user","content":"This is test 1"},
       {"role":"ai","content":"AI testing 1"},
       {"role":"user","content":"This is test 2"},
       {"role":"ai","content":"AI testing 2"},
       {"role":"user","content":"This is test 3"},
       ]

target="""<s>
\t[INST]
\t\t<<SYS>>
\t\t\tThis is system_messge_1
\t\t<</SYS>>
\t\t<<SYS>>
\t\t\tThis is system_messge_2
\t\t<</SYS>>
\t\t<<SYS>>
\t\t\tThis is system_messge_3
\t\t<</SYS>>
\t\tThis is test 1
\t[/INST]
\n\tAI testing 1\n
</s>
<s>
\t[INST]
\t\tThis is test 2
\t[/INST]
\n\tAI testing 2\n
</s>
<s>
\t[INST]
\t\tThis is test 3
\t[/INST]
\n\t"""

output = chat_list_to_INST(input)
print(target==output)

# import difflib

# def find_differences(target, output):
#     diff = difflib.ndiff(target.splitlines(1), output.splitlines(1))
#     return ''.join(diff)

# print(find_differences(target, output))
