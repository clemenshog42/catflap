import os
import glob

folder = r'C:\Public\Studium\Bachelorarbeit\code\src\ablation_studies'
files = glob.glob(os.path.join(folder, 'ablation_*.py'))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'import sys' not in content:
        new_content = "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n" + content
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
print('Fixed imports in ablation scripts!')
