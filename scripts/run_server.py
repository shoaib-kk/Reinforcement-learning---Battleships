"""Console-free server launcher.

Run with pythonw.exe (no console window) so the process can never receive
a console CTRL_C/close event — detached servers on this machine were being
killed with STATUS_CONTROL_C_EXIT. stdout/stderr go to runs/server.log.

    pythonw.exe scripts/run_server.py
"""

import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "runs", "server.log")

os.makedirs(os.path.dirname(LOG), exist_ok=True)
log = open(LOG, "a", buffering=1, encoding="utf-8", errors="replace")
sys.stdout = log
sys.stderr = log
print(f"--- server launcher start {datetime.datetime.now().isoformat()} ---")

sys.path.insert(0, ROOT)

import uvicorn  # noqa: E402  (after stdio redirect so its loggers bind to the file)

uvicorn.run("server.main:app", host="127.0.0.1", port=8000, log_level="info")
