with open('analysen.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'Ablationsstudie: Kontrastverst' in line:
        print("".join(lines[i:i+30]))
        break
