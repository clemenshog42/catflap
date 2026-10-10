import os

tex_file = r'C:\Public\Studium\Bachelorarbeit\thesis\analysen.tex'

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Only add it if it's not already there
if 'Ablationsstudie: 1-Stage (Objektdetektor) vs. 2-Stage' not in content:
    new_section = r"""

\subsection{Ablationsstudie: 1-Stage (Objektdetektor) vs. 2-Stage (Pipeline)}
Um die Architektur unserer 2-Stage-Pipeline (Face-Detektor + ResNet/MobileNet-Klassifikator) zu validieren, wurde in einer Ablationsstudie ein reiner 1-Stage-Objektdetektor (YOLOv11n) auf die Erkennung von Katzen mit und ohne Beute trainiert. Der Datensatz wurde automatisch durch das Baseline-System generiert und exakt 50:50 balanciert, wobei die Bounding-Box f\"ur Beute-Klassen um das asymmetrische Padding erweitert wurde.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/1stage_ablation_results.png}
    \caption{Trainingsmetriken des 1-Stage-Objektdetektors.}
    \label{fig:ablation_1stage_train}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{images/1stage_ablation_confusionmatrix.png}
    \caption{Konfusionsmatrix des 1-Stage-Objektdetektors auf den Validierungsdaten.}
    \label{fig:ablation_1stage_conf}
\end{figure}

Um die Modelle unter realen Bedingungen zu vergleichen, wurde der Zustandsautomat der Klappe deaktiviert (\texttt{history\_length=1}) und beide Systeme wurden auf den Testvideos mit einer Konfidenzschwelle von 0,5 evaluiert. Dabei wurde nicht nur gemessen, wie viele Fehlalarme und \"ubersehene Beute (Prey Error) auftraten, sondern auch, wie oft die Katze an sich komplett vom jeweiligen Detektor \"ubersehen wurde (Missed Cat).

\begin{figure}[H]
    \centering
    \includegraphics[width=1.0\textwidth]{images/ablation_1stage_evaluation.png}
    \caption{End-to-End-Vergleich: Klassifikationsfehler (oben) und komplett \"ubersehene Katzen (unten) pro 100 Frames.}
    \label{fig:ablation_1stage_eval}
\end{figure}

Die Ergebnisse (siehe Abbildung \ref{fig:ablation_1stage_eval}) verdeutlichen die Vor- und Nachteile der zweistufigen Architektur gegen\"uber dem direkten Ansatz.
"""
    
    # We append it right before the end of the file, or just at the very end
    content += new_section
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Ablation section appended to analysen.tex')
else:
    print('Ablation section already exists!')
