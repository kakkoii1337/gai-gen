import re

# Corrected regex pattern
TOOLS_TYPE_PREFIX_RE = r'\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'


# Test 1
test_str = ' { "type": "function", "function": { '
match = re.search(TOOLS_TYPE_PREFIX_RE, test_str)
assert(match)

# Test 2
test_str = ' { type: "function", function: { '
match = re.search(TOOLS_TYPE_PREFIX_RE, test_str)
assert(match)


