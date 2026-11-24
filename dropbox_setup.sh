#!/usr/bin/env bash
# Connects the cat flap to Dropbox: recorded clips are then uploaded to Apps/<your app name>/catflap/.
# Run on the Raspberry Pi inside the catflap folder:   bash dropbox_setup.sh
# Guide: docs/DROPBOX_SETUP.md
set -euo pipefail

cd "$(dirname "$0")"
ENV_FILE="$(pwd)/dropbox.env"

echo "=== Catflap Dropbox setup ==="
echo
echo "Step 1: Create a Dropbox app (only once)"
echo "  1. Open https://www.dropbox.com/developers/apps and click 'Create app'"
echo "  2. Choose 'Scoped access' and 'App folder', give it a name (e.g. catflap-clips)"
echo "  3. Tab 'Permissions': tick 'files.content.write', then click 'Submit'"
echo "  4. Tab 'Settings': copy the 'App key' and 'App secret'"
echo
read -r -p "App key: " APP_KEY
read -r -s -p "App secret (input hidden): " APP_SECRET
echo
if [ -z "$APP_KEY" ] || [ -z "$APP_SECRET" ]; then
    echo "App key and secret are required." >&2
    exit 1
fi

echo
echo "Step 2: Allow access"
echo "  Open this link in a browser where you are logged in to Dropbox, click 'Allow' and copy the code:"
echo
echo "  https://www.dropbox.com/oauth2/authorize?client_id=${APP_KEY}&response_type=code&token_access_type=offline"
echo
read -r -p "Code: " AUTH_CODE

RESPONSE=$(curl -s https://api.dropboxapi.com/oauth2/token \
    -d code="$AUTH_CODE" -d grant_type=authorization_code \
    -u "${APP_KEY}:${APP_SECRET}")
REFRESH_TOKEN=$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("refresh_token",""))' 2>/dev/null || true)
ACCESS_TOKEN=$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))' 2>/dev/null || true)

if [ -z "$REFRESH_TOKEN" ]; then
    echo "Dropbox did not return a token. Answer was:" >&2
    echo "$RESPONSE" >&2
    echo "Codes are valid only once and for a few minutes. Run the script again and use a fresh code." >&2
    exit 1
fi

umask 077
cat > "$ENV_FILE" <<EOF
DROPBOX_APP_KEY=${APP_KEY}
DROPBOX_APP_SECRET=${APP_SECRET}
DROPBOX_REFRESH_TOKEN=${REFRESH_TOKEN}
EOF
chmod 600 "$ENV_FILE"
echo
echo "Saved credentials to $ENV_FILE (readable only by $(whoami), not tracked by git)."

echo
echo "Step 3: Test upload"
TEST_NAME="setup_test_$(date +%Y%m%d_%H%M%S).txt"
UPLOAD=$(printf 'Catflap Dropbox test %s\n' "$(date)" | curl -s -X POST https://content.dropboxapi.com/2/files/upload \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    -H "Dropbox-API-Arg: {\"path\": \"/catflap/${TEST_NAME}\", \"mode\": \"overwrite\"}" \
    --data-binary @-)
if printf '%s' "$UPLOAD" | grep -q '"path_display"'; then
    echo "OK: uploaded /catflap/${TEST_NAME} (in Dropbox: Apps > <app name> > catflap)."
else
    echo "Test upload failed. Answer was:" >&2
    echo "$UPLOAD" >&2
    echo "Check that 'files.content.write' is ticked under Permissions and submitted, then run the script again." >&2
    exit 1
fi

echo
read -r -p "Restart the catflap service now so it starts uploading? [Y/n] " RESTART
if [ "${RESTART:-Y}" != "n" ] && [ "${RESTART:-Y}" != "N" ]; then
    sudo systemctl restart catflap
    sleep 5
    journalctl -u catflap -n 50 --no-pager -o cat | grep -E "\[Recorder\]|\[Dropbox\]" | tail -3 || true
    echo "Done. The web page shows the upload status under 'Recordings'."
else
    echo "Done. Uploads start after the next restart: sudo systemctl restart catflap"
fi
