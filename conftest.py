import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
# This allows tests to import `context_compiler` as a package.
sys.path.insert(0, str(Path(__file__).parent.absolute()))
