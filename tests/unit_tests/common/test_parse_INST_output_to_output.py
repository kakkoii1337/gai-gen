from gai.common.generators_utils import INST_output_to_output

input="""<s>\n\t[INST]\nTell me a one paragraph short story.\n\n</s>\n<s>\n\t[INST]\n\n\n</s>\n<s>\n\t[INST]\nOnce upon a time, there was a little girl named Lily who loved to play in the garden with her friends. One day, she found a beautiful flower and plucked it from its stem without thinking twice. Suddenly, an old witch appeared out of nowhere and cursed her for picking the flower. From that moment on, every time Lily touched something green, she turned into a frog. Despite this misfortune, Lily never lost hope and continued to explore the world around her, always looking"""

# input="""<s>
# \t[INST]
# \t\t<<SYS>>
# \t\t\tThis is system_messge_1
# \t\t<</SYS>>
# \t\t<<SYS>>
# \t\t\tThis is system_messge_2
# \t\t<</SYS>>
# \t\t<<SYS>>
# \t\t\tThis is system_messge_3
# \t\t<</SYS>>
# \t\tThis is test 1
# \t[/INST]
# \n\tAI testing 1\n
# </s>
# <s>
# \t[INST]
# \t\tThis is test 2
# \t[/INST]
# \n\tAI testing 2\n
# </s>
# <s>
# \t[INST]
# \t\tThis is test 3
# \t[/INST]
# \n\tAI testing 2\n\n"""

# target="AI testing 2"

output = INST_output_to_output(input).strip()
print(output)
#print(target==output)


# import difflib

# def find_differences(target, output):
#     diff = difflib.ndiff(target.splitlines(1), output.splitlines(1))
#     return ''.join(diff)

# print(find_differences(target, output))
