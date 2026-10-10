import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'

with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# The target content to replace
target = r"""Die Wahrheitsmatrix (\emph{engl. Confusion Matrix}) der Objekterkennung (Abbildung \ref{fig:confusion_det}) verdeutlicht die Präzision bei der Lokalisierung. Zur tiefergehenden Analyse unter verschiedenen Konfidenz-Schwellenwerten dienen die Präzision-Trefferquote-Kurve (\emph{engl. Precision-Recall-Curve}) (Abbildung \ref{fig:pr_curve_det}) sowie die F1-Maß-Kurve (Abbildung \ref{fig:f1_curve_det}).

\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/confusion_matrix.png}
        \caption{Wahrheitsmatrix (Detektor).}
        \label{fig:confusion_det}
    \end{minipage}\hfill
    \begin{minipage}{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/BoxPR_curve.png}
        \caption{Präzision-Trefferquote-Kurve (\emph{engl. Precision-Recall-Curve}).}
        \label{fig:pr_curve_det}
    \end{minipage}\hfill
    \begin{minipage}{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/BoxF1_curve.png}
        \caption{F1-Maß-Kurve.}
        \label{fig:f1_curve_det}
    \end{minipage}
\end{figure}"""

replacement = r"""Die Wahrheitsmatrix (\emph{engl. Confusion Matrix}) der Objekterkennung (Abbildung \ref{fig:confusion_det}) verdeutlicht die Präzision bei der Lokalisierung.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{images/confusion_matrix.png}
    \caption{Wahrheitsmatrix (Detektor).}
    \label{fig:confusion_det}
\end{figure}

Zur tiefergehenden Analyse der Modellgüte unter verschiedenen Konfidenz-Schwellenwerten dient die Präzision-Trefferquote-Kurve (\emph{engl. Precision-Recall-Curve}), die in Abbildung \ref{fig:pr_curve_det} dargestellt ist. Sie zeigt das Verhältnis zwischen richtig erkannten Objekten und Fehlalarmen.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{images/BoxPR_curve.png}
    \caption{Präzision-Trefferquote-Kurve (\emph{engl. Precision-Recall-Curve}).}
    \label{fig:pr_curve_det}
\end{figure}

Zusätzlich veranschaulicht die F1-Maß-Kurve (Abbildung \ref{fig:f1_curve_det}) den harmonischen Mittelwert aus Präzision und Trefferquote über alle Konfidenzwerte hinweg. Dies hilft dabei, den optimalen Schwellenwert für das System zu identifizieren.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{images/BoxF1_curve.png}
    \caption{F1-Maß-Kurve.}
    \label{fig:f1_curve_det}
\end{figure}"""

new_text = text.replace(target, replacement)

if text != new_text:
    with io.open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("Replace success.")
else:
    print("Error: Target not found.")
