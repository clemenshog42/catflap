bad = b'\xc3\x83\xc5\xb8'
try:
    print("Test 1:", bad.decode('utf-8').encode('cp1252').decode('utf-8'))
except Exception as e:
    print("Error 1:", e)

bad2 = b'\xc3\xa2\xe2\x82\xac\xc5\xbe'
try:
    print("Test 2:", bad2.decode('utf-8').encode('cp1252').decode('utf-8'))
except Exception as e:
    print("Error 2:", e)
