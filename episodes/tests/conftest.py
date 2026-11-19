import sys
from pathlib import Path

# The episode scripts are run as plain files, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
