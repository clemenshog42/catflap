import json

transcript_path = r'C:\Users\cleho\.gemini\antigravity\brain\aae8b8f7-ac53-4e71-bba4-ad90a88004a8\.system_generated\logs\transcript_full.jsonl'
output_path = r'c:\Public\Studium\Bachelorarbeit\code\backup.txt'

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    try:
        data = json.loads(line)
        content = data.get('content', '')
        if 'The command exited with code 0' in content and 'Einordnung und Edge-Computing' in content:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(content)
    except:
        pass
