import glob
import os

target_files = glob.glob(r'C:\Public\Studium\Bachelorarbeit\code\src\ablation_studies\generate_*.py') + \
               glob.glob(r'C:\Public\Studium\Bachelorarbeit\code\src\ablation_studies\plot_*.py')

for file_path in target_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    # Text in plots
    content = content.replace('(pro 100 Frames)', '(pro 100 Bilder)')
    content = content.replace('2-Stage', '2-stufig')
    content = content.replace('1-Stage', '1-stufig')
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {os.path.basename(file_path)}")
