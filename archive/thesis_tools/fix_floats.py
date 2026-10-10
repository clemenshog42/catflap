import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

new_text = re.sub(r'\\begin{figure}\[htbp\]', r'\\begin{figure}[H]', text)
new_text = re.sub(r'\\begin{table}\[htbp\]', r'\\begin{table}[H]', new_text)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)
print('Replaced htbp with H in analysen.tex')
