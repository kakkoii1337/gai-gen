import re

# Corrected regex pattern
TOOLNAME_RE = r'(\")?name(\")?\s*:\s*\"(.*?)\",'
target = ' { "name": "gg", "parameters": { "search_query": "latest news singapore" }'
match = re.search(TOOLNAME_RE, target)
tool_name = match.group(3)
print('name:',tool_name)

PARAMETERS_RE = r'"parameters":(\s*\{[\s\S]*?\})'
target = ' { "name": "gg", "parameters": { "search_query": "latest news singapore" }'
match = re.search(PARAMETERS_RE, target)
argument = match.group(1).strip()
print('arguments:',argument)

