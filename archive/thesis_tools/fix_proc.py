import io

file_path = 'src/processor.py'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = 'classifier_path="models/prey_V5.tflite"'
replacement = 'classifier_path="models/prey_V5_float16.tflite"'

new_text = text.replace(target, replacement)
with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Updated processor.py model path')
