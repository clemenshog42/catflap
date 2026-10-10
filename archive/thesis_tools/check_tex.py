with open('analysen.tex', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
for i, line in enumerate(lines[30:45]):
    print(f"{i+31}: {line.strip()}")
