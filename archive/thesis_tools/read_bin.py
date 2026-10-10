with open('analysen.tex', 'rb') as f:
    lines = f.readlines()
for i, line in enumerate(lines[35:45]):
    print(f"Line {i+36}: {line}")
