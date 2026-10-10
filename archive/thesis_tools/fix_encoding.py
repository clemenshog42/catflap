replacements = {
    b'\xc3\x83\xc5\xb8': 'ß'.encode('utf-8'),
    b'\xc3\x83\xe2\x80\x9e': 'Ä'.encode('utf-8'),
    b'\xc3\x83\xc2\xa4': 'ä'.encode('utf-8'),
    b'\xc3\x83\xc2\xb6': 'ö'.encode('utf-8'),
    b'\xc3\x83\xc2\xbc': 'ü'.encode('utf-8'),
    b'\xc3\x83\xc2\x9c': 'Ü'.encode('utf-8'),
    b'\xc3\x83\xc2\x96': 'Ö'.encode('utf-8'),
    b'\xc3\x83\xc2\x84': 'Ä'.encode('utf-8'),
    b'\xc3\xa2\xe2\x82\xac\xc5\xbe': '„'.encode('utf-8'),
    b'\xc3\xa2\xe2\x82\xac\xc2\x9d': '“'.encode('utf-8'),
    b'\xc3\xa2\xe2\x82\xac\xe2\x80\x9c': '–'.encode('utf-8'),
}

with open('analysen.tex', 'rb') as f:
    data = f.read()

for bad, good in replacements.items():
    data = data.replace(bad, good)

with open('analysen.tex', 'wb') as f:
    f.write(data)
