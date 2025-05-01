# conftest.py

import os, sys

# Insert the src/ directory into sys.path so `import src.*` works
here = os.path.dirname(__file__)
src  = os.path.join(here, "src")
sys.path.insert(0, src)
