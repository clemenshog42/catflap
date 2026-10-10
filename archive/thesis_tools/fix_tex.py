with open('analysen.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if r'\subsection{Ablationsstudie: Kontrastverst' in line or r'\subsubsection{Ablationsstudie: Kontrastverst' in line:
        new_lines.append(r'\subsubsection{Ablationsstudie: Kontrastverstärkung (CLAHE) im Klassifikator}' + '\n')
        new_lines.append(r'Um den Effekt unter realistischen Bedingungen zu isolieren, wurden die Farbbilder der Testvideos in Graustufen konvertiert, um die Infrarotaufnahme einer echten Katzenklappe zu simulieren. Die Ergebnisse (siehe Abbildung \ref{fig:ablation_clahe}) zeigen, dass der CLAHE-Filter kein universelles Wundermittel ist, sondern einen interessanten Kompromiss darstellt.' + '\n\n')
        new_lines.append(r'''\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{images/ablation_clahe_state_changes.png}
    \caption{Vergleich der normalisierten Zustandswechsel bei Videos \emph{mit Beute} mit und ohne CLAHE-Filter im Klassifikator auf Graustufenbildern.}
    \label{fig:ablation_clahe}
\end{figure}''' + '\n\n')
        new_lines.append(r'Bei den Testvideos \emph{ohne Beute} reduziert der CLAHE-Filter das unerwünschte Flackern und die Fehlalarme bei schwierigen Kontrasten signifikant (z.\,B. bei \texttt{black\_gray.mp4} von 14 auf 5 Zustandswechsel). Paradoxerweise kehrt sich dieser Effekt bei den Videos \emph{mit Beute} teilweise um: Hier war das Modell ohne künstliche Kontrastverstärkung deutlich stabiler (11 statt 19 Zustandswechsel bei \texttt{black\_gray.mp4}). Es ist davon auszugehen, dass der harte CLAHE-Filter zwar die Umrisse schwarzer Katzen besser vom Hintergrund abhebt, aber gleichzeitig die feinen Texturen der Beute im Maul der Katze überzeichnet und so die Konfidenz des Klassifikators bei der Beuteerkennung schwächt.' + '\n\n')
        new_lines.append(r'Da der nachgeschaltete Zustandsautomat (vgl. Kapitel \ref{sec:analysen_zustand}) ohnehin jegliches Flackern zuverlässig glättet, kann geschlussfolgert werden, dass der ressourcenintensive CLAHE-Filter für dieses spezifische Problem nicht zwingend erforderlich ist und das reine Infrarot-Graustufenbild für eine zuverlässige Klassifikation ausreicht.' + '\n\n')
        skip = True
    elif r'\subsection{Ablationsstudie: 1-Stage' in line or r'\subsubsection{Ablationsstudie: 1-Stage' in line:
        skip = False
        new_lines.append(r'\subsubsection{Ablationsstudie: 1-Stage (Objektdetektor) vs. 2-Stage (Pipeline)}' + '\n')
    elif not skip:
        new_lines.append(line)

with open('analysen.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
