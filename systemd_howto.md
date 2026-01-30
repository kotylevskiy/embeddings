# Running as a Linux Service (systemd)

This section describes how to run **Embeddings** as a background service on Linux using **systemd**, and how to view its logs.

Before applying this, run Embeddings at least once and confirm it works (including tests).

## 1. Create a systemd service file

Create the service definition:

```bash
sudo nano /etc/systemd/system/embeddings.service
```

Paste the following and adjust paths:

```ini
[Unit]
Description=Embeddings (FastAPI text-embeddings service)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=appuser # Adjust this
Group=appuser # Adjust this

WorkingDirectory=/path/to/embeddings  # Adjust this
EnvironmentFile=/path/to/embeddings/.env  # Adjust this

# Safe defaults
Environment="EMBEDDINGS_HOST=127.0.0.1"
Environment="EMBEDDINGS_PORT=11445"

# Adjust this
ExecStart=/path/to/embeddings/.venv/bin/uvicorn service.main:app \
  --host ${EMBEDDINGS_HOST} \
  --port ${EMBEDDINGS_PORT}

Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

**Notes**

* All paths must be **absolute**
* Replace `/path/to/embeddings` with the actual project directory
* The service runs as a non-root user (`appuser`)

## 2. Reload systemd and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now embeddings.service
```

## 3. Check service status

```bash
systemctl status embeddings.service
```

If the service fails to start:

```bash
journalctl -u embeddings.service -xe
```

## 4. Viewing Logs

The service logs are collected automatically by **systemd-journald**.

Follow logs in real time:

```bash
journalctl -u embeddings.service -f
```

View recent logs:

```bash
journalctl -u embeddings.service -n 100
```
