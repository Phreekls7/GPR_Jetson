# Hidden Graves GPR Server & Reader

## Setup

1. Create and activate a venv:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # cmd.exe
   .\.venv\Scripts\activate.bat
   ```

2. Install deps:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Server

Runs on port 5000 by default.

```bash
python server.py
```

Clients connect via Socket.IO.

- Emit `get_raw_gpr` with `{ "count": N }`  
- Server replies with `raw_gpr_data` event containing last N traces

## Client

```bash
python client.py --host 192.168.4.1 --port 5000 --count 100
```

Adjust `--host`, `--port`, and `--count` as needed.
