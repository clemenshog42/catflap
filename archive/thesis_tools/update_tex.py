import re

with open('analysen.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the subsection to subsubsection for the last two ablations which were missed by powershell
content = content.replace(r'\subsection{Ablationsstudie: Kontrastverstärkung (CLAHE) im Klassifikator}', r'\subsubsection{Ablationsstudie: Kontrastverstärkung (CLAHE) im Klassifikator}')
content = content.replace(r'\subsection{Ablationsstudie: 1-Stage (Objektdetektor) vs. 2-Stage (Pipeline)}', r'\subsubsection{Ablationsstudie: 1-Stage (Objektdetektor) vs. 2-Stage (Pipeline)}')

old_text = r'''Die Ergebnisse (siehe Abbildung \ref{fig:ablation_clahe}) zeigen, dass der Filter bei Katzen ohne Beute kaum einen Unterschied macht (die Fehlalarmquote bleibt nahezu identisch). Einen massiven Einfluss hat der Filter jedoch bei Videos \emph{mit Beute} unter schwierigen Lichtbedingungen.

Besonders beim Testvideo \texttt{black.mp4} (schwarze Katze in der Nacht, die eine Beute im Maul trägt) wird der Unterschied deutlich: Mit aktiviertem CLAHE-Filter (Baseline) erkennt das System die Beute sofort und verriegelt die Klappe mit genau 1 Zustandswechsel dauerhaft. Ohne den CLAHE-Filter (Ablation) stieg die Rate auf über 2 Zustandswechsel pro 100 Frames an (insgesamt 5 Wechsel im kurzen Clip). Das bedeutet, dass das System die Klappe verriegelte, sie kurz darauf wieder entriegelte und wieder verriegelte (ein ständiges \glqq Flackern\grqq). Der Klassifikator verlor ohne den erhöhten Kontrast mehrfach die Konfidenz in die Erkennung der Beute, was im realen Einsatz dazu führen könnte, dass die Katze im falschen Moment doch mit der Maus das Haus betritt. Der CLAHE-Filter ist somit ein kritischer Baustein für die Stabilität des Systems bei Nachtaufnahmen.'''

new_text = r'''Um den Effekt unter realistischen Bedingungen zu isolieren, wurden die Farbbilder der Testvideos in Graustufen konvertiert, um die Infrarotaufnahme einer echten Katzenklappe zu simulieren. Die Ergebnisse (siehe Abbildung \ref{fig:ablation_clahe}) zeigen, dass der CLAHE-Filter kein universelles Wundermittel ist, sondern einen interessanten Kompromiss darstellt.

Bei den Testvideos \emph{ohne Beute} reduziert der CLAHE-Filter das unerwünschte Flackern und die Fehlalarme bei schwierigen Kontrasten signifikant (z.\,B. bei \texttt{black\_gray.mp4} von 14 auf 5 Zustandswechsel). Paradoxerweise kehrt sich dieser Effekt bei den Videos \emph{mit Beute} teilweise um: Hier war das Modell ohne künstliche Kontrastverstärkung deutlich stabiler (11 statt 19 Zustandswechsel bei \texttt{black\_gray.mp4}). Es ist davon auszugehen, dass der harte CLAHE-Filter zwar die Umrisse schwarzer Katzen besser vom Hintergrund abhebt, aber gleichzeitig die feinen Texturen der Beute im Maul der Katze überzeichnet und so die Konfidenz des Klassifikators bei der Beuteerkennung schwächt.

Da der nachgeschaltete Zustandsautomat (vgl. Kapitel \ref{sec:analysen_zustand}) ohnehin jegliches Flackern zuverlässig glättet, kann geschlussfolgert werden, dass der ressourcenintensive CLAHE-Filter für dieses spezifische Problem nicht zwingend erforderlich ist und das reine Infrarot-Graustufenbild für eine zuverlässige Klassifikation ausreicht.'''

content = content.replace(old_text, new_text)

with open('analysen.tex', 'w', encoding='utf-8') as f:
    f.write(content)
