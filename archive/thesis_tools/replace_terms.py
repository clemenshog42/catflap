import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'

with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Make replacements
# Confusion Matrix
text = text.replace('Die Confusion Matrix der Objekterkennung', 'Die Wahrheitsmatrix (\\emph{engl. Confusion Matrix}) der Objekterkennung')
text = text.replace('Confusion Matrix der Systemevaluation', 'Wahrheitsmatrix der Systemevaluation')
text = text.replace('folgenden Confusion Matrix dargestellt', 'folgenden Wahrheitsmatrix dargestellt')
text = text.replace('Confusion Matrix (Detektor)', 'Wahrheitsmatrix (Detektor)')
text = text.replace('Confusion Matrix (Klassifikator)', 'Wahrheitsmatrix (Klassifikator)')

# Loss
text = text.replace('sinkenden Loss', 'sinkenden Trainingsfehler (\\emph{engl. Loss})')
text = text.replace('der Validation Loss', 'der Validierungsfehler')

# Precision / Recall
text = text.replace('Precision-Recall-Kurve', 'Präzision-Trefferquote-Kurve (\\emph{engl. Precision-Recall-Curve})')
text = text.replace('Pr\\äzision (\\emph{Precision})', 'Präzision') # Cleanup if already present
text = text.replace('Precision (', 'Präzision (')
text = text.replace('Recall (', 'Trefferquote (')
text = text.replace('Vorhersagegenauigkeit (Genauigkeit)', 'Präzision (\\emph{engl. Precision})')
text = text.replace('Trefferquote (Sensitivität)', 'Trefferquote (\\emph{engl. Recall})')
text = text.replace('Vorhersagegenauigkeits-Kompromiss', 'Präzisions-Trefferquoten-Kompromiss')
text = text.replace('Vorhersagegenauigkeit', 'Präzision') # Generic replacements afterwards

# TP / TN / FP / FN
text = text.replace('False Negative', 'Falsch-Negativ-Ergebnis (\\emph{engl. False Negative})')
text = text.replace('False Positives', 'Falsch-Positiven (Fehlalarmen)')
text = text.replace('False-Positive-Rate', 'Fehlalarmquote')
text = text.replace('False-Positive-Testvideos', 'Testvideos mit potenziellen Fehlalarmen')

# Other
text = text.replace('\\glqq moving average\\grqq', '\\glqq gleitendem Durchschnitt\\grqq\\ (\\emph{engl. moving average})')
text = text.replace('asymmetrischen Croppings', 'asymmetrischen Bildzuschnitts')
text = text.replace('symmetrischem Cropping', 'symmetrischem Bildzuschnitt')
text = text.replace('symmetrisches Cropping', 'symmetrischer Bildzuschnitt')

# F1
text = text.replace('F1-Kurve', 'F1-Maß-Kurve')

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Replacements successful.")
