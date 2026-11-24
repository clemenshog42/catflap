# Dropbox upload of the recorded clips

The Pi server saves one video clip per detection episode in `recordings/`. Each clip is the raw camera
frames, with a `.json` file holding the decision and the events. Once Dropbox is set up, every finished
clip is uploaded to **Dropbox > Apps > <app name> > catflap**. The local copy stays on the Pi; the oldest
local clips are deleted above `--record-max-gb` (default 5 GB).

## Setup (about 5 minutes)

1. **Create a Dropbox app**
   - Open <https://www.dropbox.com/developers/apps> and click **Create app**.
   - Choose **Scoped access** and **App folder**. Name it, e.g. `catflap-clips`.
   - In the **Permissions** tab, tick `files.content.write` and click **Submit**.
   - In the **Settings** tab, find the **App key** and **App secret**.

2. **Run the setup script on the Pi**
   ```bash
   ssh pi@100.73.232.80
   cd ~/catflap
   bash dropbox_setup.sh
   ```
   The script:
   1. asks for the app key and secret;
   2. shows a link: open it, click **Allow**, and paste the code;
   3. saves the credentials to `~/catflap/dropbox.env` (readable only by `pi`, not in git);
   4. uploads a small test file to `/catflap/`;
   5. offers to restart the service (asks for the `sudo` password).

3. **Check**: the web page shows `Dropbox: ready` under *Recordings*, or `last upload HH:MM` after the
   first clip.

## Notes
- The app only has access to its own folder (`Apps/<app name>`), not to the rest of your Dropbox.
- The refresh token does not expire. To revoke access, remove the app at
  <https://www.dropbox.com/account/connected_apps>, then delete `dropbox.env`.
- Failed uploads are retried 5 times (after 30 s, 60 s, ...). Log lines start with `[Dropbox]`:
  `journalctl -u catflap | grep Dropbox`
- Space: a 1-minute clip is about 10–20 MB.
