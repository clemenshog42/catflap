#!/usr/bin/env bash

SERVICE_NAME="catflap"
USER_NAME="$USER"  # user running the service

# Automatically use the directory where the script is executed
WORKING_DIR="$(pwd)"  
VENV_DIR="$WORKING_DIR/venv"
PYTHON_SCRIPT="src/pi_server.py"

# -------------------------------
# INSTALL SYSTEM DEPENDENCIES
# -------------------------------
echo "[1/5] Installing system packages..."

sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-venv python3-pip python3-opencv

# -------------------------------
# CREATE VENV WITH SYSTEM PACKAGES
# -------------------------------
echo "[2/5] Creating venv with system-site-packages..."
# --system-site-packages is CRITICAL so the venv can see python3-picamera2!
python3 -m venv --system-site-packages "$VENV_DIR"

# Upgrade pip inside venv
"$VENV_DIR/bin/pip" install --upgrade pip

echo "[3/5] Installing requirements..."

# Create a local temporary directory to prevent /tmp (RAM disk) from filling up
mkdir -p "$WORKING_DIR/pip_tmp"
export TMPDIR="$WORKING_DIR/pip_tmp"

# Pre-install CPU-only PyTorch to prevent pip from downloading massive NVIDIA CUDA wheels
echo "Installing CPU-only PyTorch..."
"$VENV_DIR/bin/pip" install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

if [ -f "$WORKING_DIR/pi_requirements.txt" ]; then
    echo "Installing remaining packages from pi_requirements.txt..."
    "$VENV_DIR/bin/pip" install --no-cache-dir -r "$WORKING_DIR/pi_requirements.txt"
else
    echo "Warning: pi_requirements.txt not found in $WORKING_DIR"
fi

# Cleanup temp dir
rm -rf "$WORKING_DIR/pip_tmp"
unset TMPDIR

# -------------------------------
# CREATE SYSTEMD SERVICE
# -------------------------------
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "[4/5] Creating systemd service at $SERVICE_PATH..."

sudo bash -c "cat > $SERVICE_PATH" << EOF
[Unit]
Description=Catflap Picamera2 Flask Server
After=network.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${WORKING_DIR}

# Use venv python
ExecStart=${VENV_DIR}/bin/python ${WORKING_DIR}/${PYTHON_SCRIPT}

Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# -------------------------------
# ENABLE + START SERVICE
# -------------------------------
echo "[5/5] Reloading systemd, enabling, and starting service..."

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl start "${SERVICE_NAME}"

echo "--------------------------------------"
echo "Systemd service '${SERVICE_NAME}' installed and started."
echo "The Flask Server is now running in the background!"
echo "View stream at: http://<YOUR_PI_IP>:5000"
echo "View Swagger:   http://<YOUR_PI_IP>:5000/apidocs"
echo ""
echo "Check logs using:"
echo " sudo journalctl -u ${SERVICE_NAME} -f"
echo "--------------------------------------"
