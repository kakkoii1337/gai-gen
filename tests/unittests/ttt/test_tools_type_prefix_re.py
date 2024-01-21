import re

# Corrected regex pattern
TEXT_TYPE_PREFIX_RE = r'\s*{\s*(\")?type(\")?\s*:\s*"text",\s*(\")?text(\")?\s*:\s*"'


# Test 1
test_str = ' { type: "text", text: "'
match = re.search(TEXT_TYPE_PREFIX_RE, test_str)
assert(match)

# Test 2
test_str = ' {\n    "type": "text",\n    "text": "'
match = re.search(TEXT_TYPE_PREFIX_RE, test_str)
assert(match)

