import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

new_text = text.replace(r"\glqq Keine Beute\grqq\.", r"\glqq Keine Beute\grqq.")

if new_text != text:
    with io.open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Fixed runaway argument error.')
else:
    print('Error: Target not found.')
