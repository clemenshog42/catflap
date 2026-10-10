import re
import glob

for tex_file in glob.glob('C:/Public/Studium/Bachelorarbeit/code/*.tex'):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We want to find unescaped underscores outside of labels, urls, etc.
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '_' in line and '\_' not in line:
            # ignore label, cite, url, ref, includegraphics, include, input, begin, end, lstlisting, math mode
            if not any(x in line for x in ['label', 'cite', 'url', 'ref', 'includegraphics', 'include', 'input', 'begin', 'end', 'lstlisting', '$', '\\_']):
                print(f'{tex_file}:{i+1}: {line}')
