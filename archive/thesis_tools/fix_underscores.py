import io

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/implementierung.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"Außerdem bietet die Klasse Methoden zum schließen \texttt{lock_in} / \texttt{lock_out} und öffnen \texttt{unlock_in} / \texttt{unlock_out} der Klappe."
replacement = r"Außerdem bietet die Klasse Methoden zum schließen \texttt{lock\_in} / \texttt{lock\_out} und öffnen \texttt{unlock\_in} / \texttt{unlock\_out} der Klappe."

new_text = text.replace(target, replacement)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Fixed underscores in implementierung.tex!')
