with open('analysen.tex', 'rb') as f:
    raw_data = f.read()

# Replace any corrupted byte sequences that correspond to smart quotes
# „ in utf8 is \xe2\x80\x9e
# “ in utf8 is \xe2\x80\x9c
# If they got mangled, let's just decode with replace and then fix it
text = raw_data.decode('utf-8', errors='replace')
text = text.replace('', '"')
text = text.replace('„', '"')
text = text.replace('“', '"')
text = text.replace('\x9d', '"')
text = text.replace('â€ž', '"')
text = text.replace('â€œ', '"')

with open('analysen.tex', 'w', encoding='utf-8') as f:
    f.write(text)
print("File cleaned.")
