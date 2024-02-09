import re

# Corrected regex pattern
JSON_SUFFIX_RE = r'\s*"\s*}\s*$'

# Test 1
test_str = 'Alice in wonderland." }'
replaced = re.sub(JSON_SUFFIX_RE, '',test_str)
assert(replaced == 'Alice in wonderland.')

# Test 2
test_str = 'Alice in wonderland.\n"}'
replaced = re.sub(JSON_SUFFIX_RE, '',test_str)
assert(replaced == 'Alice in wonderland.')
