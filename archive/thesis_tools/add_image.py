import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/confusion_matrix_cls.png}
    \caption{Confusion Matrix des Klassifikators auf dem Validierungsdatensatz.}
    \label{fig:confusion_cls}
\end{figure}"""

replacement = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/confusion_matrix_cls.png}
    \caption{Confusion Matrix des Klassifikators auf dem Validierungsdatensatz.}
    \label{fig:confusion_cls}
\end{figure}

Um einen visuellen Eindruck der Klassifikationsergebnisse zu vermitteln, zeigt Abbildung \ref{fig:val_pred_cls} einen beispielhaften Batch an Vorhersagen des Klassifikators auf dem Validierungsdatensatz. Wie man gut erkennen kann, lernt das Modell die entscheidenden Merkmale im Bereich des Katzenmauls extrem zuverlässig aus den asymmetrisch zugeschnittenen Bildbereichen.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/val_batch0_pred_cls.jpg}
    \caption{Beispielhafte Vorhersagen des Klassifikators auf dem Validierungsdatensatz.}
    \label{fig:val_pred_cls}
\end{figure}"""

new_text = text.replace(target, replacement)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Image added to analysen.tex!')
