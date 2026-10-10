import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"""\begin{itemize}
    \item \textbf{True Positive (TP):} Katze mit Beute darf nicht ins Haus.
    \item \textbf{Falsch-Negativ-Ergebnis (\emph{engl. False Negative}) (FN):} Katze mit Beute darf ins Haus.
    \item \textbf{Korrekte Abweisung (TN):} Katze ohne Beute darf ins Haus.
    \item \textbf{Fehlalarm (FP):} Katze ohne Beute darf nicht ins Haus.
\end{itemize}"""

replacement = r"""\begin{itemize}
    \item \textbf{Richtig-Positiv-Ergebnis (\emph{engl. True Positive}) (TP):} Katze mit Beute darf nicht ins Haus (korrekte Sperrung).
    \item \textbf{Falsch-Negativ-Ergebnis (\emph{engl. False Negative}) (FN):} Katze mit Beute darf ins Haus (Beute gelangt fälschlicherweise hinein).
    \item \textbf{Richtig-Negativ-Ergebnis (\emph{engl. True Negative}) (TN):} Katze ohne Beute darf ins Haus (korrekter Durchlass).
    \item \textbf{Falsch-Positiv-Ergebnis (\emph{engl. False Positive}) (FP):} Katze ohne Beute darf nicht ins Haus (unnötiger Fehlalarm).
\end{itemize}"""

new_text = text.replace(target, replacement)

# We also need to fix the paragraph right below that mentions "Ein Fehlalarm bedeutet..." or similar
new_text = new_text.replace(
    r"Ein Fehlalarm bedeutet, dass das Haustier grundlos vor einer verschlossenen Tür steht.",
    r"Ein Falsch-Positiv-Ergebnis (Fehlalarm) bedeutet, dass das Haustier grundlos vor einer verschlossenen Tür steht."
)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Updated terms in analysen.tex!')
