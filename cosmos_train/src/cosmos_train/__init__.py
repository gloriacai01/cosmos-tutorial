"""Dataset preparation tools for Cosmos Policy experiments."""

import os

# Apply before any submodule can import Prefect. DO_NOT_TRACK disables SDK
# analytics; the Prefect setting disables the local ephemeral/server heartbeat.
os.environ["DO_NOT_TRACK"] = "1"
os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
