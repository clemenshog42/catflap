import os
import glob
import json
import shutil

history_dir = r"C:\Users\cleho\AppData\Roaming\Code\User\History"
project_dir = r"c:\Public\Studium\Bachelorarbeit"
tex_files = [
    "einleitung.tex", "grundlagen.tex", "material.tex", "methodik.tex",
    "implementierung.tex", "analysen.tex", "diskussion.tex", "zusammenfassung.tex",
    "bachelor-thesis.tex"
]

target_files = {os.path.join(project_dir, f).lower(): f for f in tex_files}

found_versions = {f: [] for f in tex_files}

for root, dirs, files in os.walk(history_dir):
    if "entries.json" in files:
        entries_path = os.path.join(root, "entries.json")
        try:
            with open(entries_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            resource = data.get("resource", "").replace("file:///", "").replace("%3A", ":").replace("/", "\\").lower()
            if resource in target_files:
                tex_name = target_files[resource]
                for entry in data.get("entries", []):
                    entry_path = os.path.join(root, entry["id"])
                    timestamp = entry.get("timestamp", 0)
                    found_versions[tex_name].append((timestamp, entry_path))
        except Exception as e:
            pass

for tex_name, versions in found_versions.items():
    if not versions:
        print(f"No history found for {tex_name}")
        continue
    # sort by timestamp desc
    versions.sort(key=lambda x: x[0], reverse=True)
    latest_path = versions[0][1]
    dest_path = os.path.join(project_dir, tex_name)
    shutil.copy2(latest_path, dest_path)
    print(f"Recovered {tex_name} from {latest_path}")
