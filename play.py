"""
Root launcher shortcut for NodeMCU Dino AI
Runs bridge/play_nodemcu.py
"""
import os
import sys

# Add bridge directory to path and execute
bridge_script = os.path.join(os.path.dirname(__file__), "bridge", "play_nodemcu.py")
if os.path.exists(bridge_script):
    import runpy
    runpy.run_path(bridge_script, run_name="__main__")
else:
    print(f"❌ Could not find {bridge_script}")
