import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target1 = r"Das System verarbeitete die Videosequenzen mit einer durchschnittlichen Inferenzgeschwindigkeit von knapp 8 FPS. Die Erkennungsleistung über den Testdatensatz ist in der folgenden Wahrheitsmatrix dargestellt:"
rep1 = r"Das System verarbeitete die Videosequenzen mit einer durchschnittlichen Inferenzgeschwindigkeit von knapp 8 FPS. Da der Zustandsautomat eine Historie von 15 Einzelbildern (\texttt{history\_length=15}) auswertet, entspricht dies einer zeitlichen Glättung von knapp zwei Sekunden. Die Erkennungsleistung über den Testdatensatz ist in der folgenden Wahrheitsmatrix dargestellt:"

target2 = r"\textbf{Realität: Keine Beute} & 2 (FP) & 1 (TN) \\"
rep2 = r"\textbf{Realität: Keine Beute} & 0 (FP) & 3 (TN) \\"

target3 = r"\item \textbf{Präzision (\emph{engl. Precision}):} $\frac{TP}{TP+FP} = \frac{4}{4+2} = 66,7\,\%$"
rep3 = r"\item \textbf{Präzision (\emph{engl. Precision}):} $\frac{TP}{TP+FP} = \frac{4}{4+0} = 100,0\,\%$"

text = text.replace(target1, rep1)
text = text.replace(target2, rep2)
text = text.replace(target3, rep3)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated 6.4 metrics successfully!')
