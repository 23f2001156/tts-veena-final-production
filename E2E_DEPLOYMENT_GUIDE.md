# E2E Networks Deployment Guide - Veena TTS Server

A beginner-friendly, step-by-step guide to deploy your TTS server on E2E Networks GPU cloud **without Docker**.

---

## Prerequisites

- **E2E Networks Account**: Sign up at [e2enetworks.com](https://www.e2enetworks.com)
- **SSH Client**: Windows Terminal, PuTTY, or Git Bash
- **Project Files**: Your local `tts veena production` folder

---

## Step 1: Generate SSH Key (on Your PC)

Open **PowerShell** or **Git Bash** and run:

```bash
ssh-keygen -t rsa -b 4096
```

- Press Enter to save to default location (`C:\Users\YourName\.ssh\id_rsa`)
- Optionally set a passphrase (or leave empty)
- This creates two files:
  - `id_rsa` (private key - keep secret!)
  - `id_rsa.pub` (public key - share with E2E)

**Copy your public key:**
```bash
cat ~/.ssh/id_rsa.pub
```
Copy the entire output (starts with `ssh-rsa`).

---

## Step 2: Add SSH Key to E2E Networks

1. Log in to [E2E Cloud Portal](https://myaccount.e2enetworks.com)
2. Go to **My Account** → **SSH Keys**
3. Click **Add SSH Key**
4. Paste your public key and give it a name
5. Save

---

## Step 3: Create a GPU Instance

1. Go to **Compute** → **Nodes** → **Create Node**
2. Select an **image**:
   - Choose **Ubuntu 22.04** with **CUDA** pre-installed (recommended)
   - Or select a PyTorch/ML image if available
3. Select **GPU Plan**:
   - Start with lowest tier to test (T4, L4, or A10)
   - Scale up later based on performance needs
4. **Instance Name**: `tts-veena-server`
5. **Storage**: 100 GB minimum (model downloads ~15GB)
6. **SSH Key**: Select the key you added in Step 2
7. **Security Group**: 
   - Ensure ports **22** (SSH) and **8001** (TTS) are open
   - Or create a new security group with these ports
8. Click **Create** and wait for the instance to start

---

## Step 4: Connect to Your Server

Once the instance shows **Running**, note the **Public IP**.

**Connect via SSH:**
```bash
ssh root@YOUR_PUBLIC_IP
```

Example:
```bash
ssh root@49.50.xx.xx
```

First connection will ask to confirm fingerprint - type `yes`.

---

## Step 5: Initial Server Setup

Run these commands on your E2E server:

```bash
# Update system
apt update && apt upgrade -y

# Install Python 3.11 (if not already installed)
apt install python3.11 python3.11-venv python3-pip -y

# Verify CUDA is working
nvidia-smi
```

You should see your GPU listed. If not, contact E2E support.

---

## Step 6: Upload Your Project Files

**From your local PC (PowerShell/Git Bash):**

```bash
# Navigate to your project folder
cd "C:\Users\aqdas\Desktop\final models\tts veena production"

# Upload entire folder to server
scp -r . root@YOUR_PUBLIC_IP:/root/tts-server/
```

This copies all your files to `/root/tts-server/` on the server.

---

## Step 7: Install Dependencies

**On the E2E server:**

```bash
# Go to project folder
cd /root/tts-server

# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

> **Note**: First run will download the Veena model (~15GB). Be patient!

---

## Step 8: Test the TTS Server

**Start the server manually first to check for errors:**

```bash
cd /root/tts-server
source venv/bin/activate

# Run server
python tts_server.py
```

You should see:
```
INFO - Loading Veena TTS model...
INFO - Loading SNAC decoder...
INFO - Models loaded in XX.XXs
INFO - Server ready. Max concurrent requests: 10
```

**Test from your PC:**
```bash
curl -X POST "http://YOUR_PUBLIC_IP:8001/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, यह एक परीक्षण है।", "speaker": "kavya"}' \
  --output test.pcm
```

If you get audio data, it's working! Press `Ctrl+C` to stop the server.

---

## Step 9: Run as Background Service

Create a **systemd service** to keep the server running:

```bash
# Create service file
cat > /etc/systemd/system/tts-server.service << 'EOF'
[Unit]
Description=Veena TTS Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/tts-server
Environment="PATH=/root/tts-server/venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/root/tts-server/venv/bin/python -m uvicorn tts_server:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

**Enable and start the service:**

```bash
# Reload systemd
systemctl daemon-reload

# Enable on boot
systemctl enable tts-server

# Start now
systemctl start tts-server

# Check status
systemctl status tts-server
```

---

## Step 10: Verify Deployment

**Check server is running:**
```bash
curl http://localhost:8001/health
```

Expected output:
```json
{
  "status": "healthy",
  "model": "maya-research/Veena",
  "speakers": ["kavya", "agastya", "maitri", "vinaya"],
  "sample_rate": 24000,
  "max_concurrent": 10,
  "gpu_available": true
}
```

---

## Useful Commands

| Action | Command |
|--------|---------|
| View logs | `journalctl -u tts-server -f` |
| Restart server | `systemctl restart tts-server` |
| Stop server | `systemctl stop tts-server` |
| Check GPU usage | `nvidia-smi` |
| Check memory | `free -h` |

---

## Troubleshooting

### "CUDA out of memory"
- Your GPU may be too small for the model
- Try a larger GPU plan on E2E

### "Connection refused" on port 8001
- Check security group allows port 8001
- Verify server is running: `systemctl status tts-server`

### Model download fails
- Check internet: `ping google.com`
- Manual download: `huggingface-cli download maya-research/Veena`

### "Module not found" errors
- Ensure venv is activated: `source /root/tts-server/venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

---

## Your TTS Endpoint

After deployment, your TTS service will be available at:

```
http://YOUR_PUBLIC_IP:8001/tts/synthesize
```

**API Documentation**: `http://YOUR_PUBLIC_IP:8001/docs`

---

## Cost Tips

- **Stop instance** when not in use (you're billed by hour)
- **Use reserved instances** for long-term use (cheaper)
- **Monitor usage** in E2E dashboard to avoid surprises
