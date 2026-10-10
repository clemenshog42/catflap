import re
import glob

for tex_file in glob.glob('C:/Public/Studium/Bachelorarbeit/code/*.tex'):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all occurrences of unescaped underscores in the text
    # We ignore images, labels, cites, urls which have _ safely.
    # Actually, a quick way is to just find \texttt{...} containing an unescaped underscore
    matches = re.findall(r'\\texttt\{([^\}]+)\}', content)
    for match in matches:
        if '_' in match and r'\_' not in match:
            print(f'File {tex_file} has unescaped underscore in texttt: {match}')
