import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Update 6.7
text = re.sub(
    r"\\subsection\{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens\}.*?(?=\\subsection\{Ablationsstudie: Kontrastverst)",
    r"""\\subsection{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens}
Um die isolierte Wirkung des asymmetrischen Zuschneidens (Verschiebung der Bounding-Box nach unten, um den typischen Ort der Beute zu fokussieren) zu untersuchen, wurde in dieser Studie der glättende Zustandsautomat explizit deaktiviert (\\texttt{history\\_length=1}). Nur so werden die rohen Unsicherheiten des Klassifikators sichtbar.

Wie in Abbildung \\ref{fig:ablation_crop} zu sehen ist, offenbart sich hierbei ein faszinierender Kompromiss (\\emph{engl. Trade-off}): 
Das asymmetrische Zuschneiden macht das System im unteren Bildbereich deutlich empfindlicher. In reinen False-Positive-Videos wie \\texttt{two\\_cats.mp4} führt dies zu einer höheren Fehlerquote (Anstieg von 0,93 auf 3,01 Zustandswechsel pro 100 Frames), da das Modell durch das asymmetrische Raster häufiger auf Halsbänder oder Pfoten im unteren Rand fokussiert.

\\begin{figure}[H]
    \\centering
    \\includegraphics[width=0.7\\textwidth]{images/ablation_crop_state_changes.png}
    \\caption{Isolierte Auswirkung des asymmetrischen Croppings auf die rohe Systemstabilität (ohne Zustandsautomat).}
    \\label{fig:ablation_crop}
\\end{figure}

Gleichzeitig rettet genau diese Empfindlichkeit die Erkennung von echter Beute: Im dunklen Video \\texttt{black.mp4} (Kategorie \\emph{with\\_prey}) senkt das asymmetrische Cropping die Fehlerquote drastisch von 3,75 auf winzige 0,42 Zustandswechsel pro 100 Frames. Das Modell "übersieht" die Beute mit asymmetrischem Zuschnitt also kaum noch.
Da der nachgelagerte Zustandsautomat (wie in Kapitel 6.6 bewiesen) die False-Positives exzellent herausfiltert, ist das asymmetrische Zuschneiden ein massiver Gewinn für das Gesamtsystem, da es die kritische Erkennung der Beute absichert.

""",
    text,
    flags=re.DOTALL
)

# Update 6.8
text = re.sub(
    r"\\subsection\{Ablationsstudie: Kontrastverstärkung \(CLAHE\) im Klassifikator\}.*?(?=\\section\{Diskussion)",
    r"""\\subsection{Ablationsstudie: Kontrastverstärkung (CLAHE) im Klassifikator}
Auch die isolierte Auswirkung des CLAHE-Filters auf den Klassifikator wurde ohne Zustandsautomat (\\texttt{history\\_length=1}) untersucht, um zu bewerten, inwiefern der adaptive Kontrast dem KI-Modell in Extremsituationen hilft.

\\begin{figure}[H]
    \\centering
    \\includegraphics[width=0.7\\textwidth]{images/ablation_clahe_state_changes.png}
    \\caption{Isolierte Auswirkung des CLAHE-Filters auf die rohe Systemstabilität in dunklen Szenen (ohne Zustandsautomat).}
    \\label{fig:ablation_clahe}
\\end{figure}

Die Ergebnisse (siehe Abbildung \\ref{fig:ablation_clahe}) spiegeln ein ähnliches Verhalten wie das asymmetrische Zuschneiden wider: Der CLAHE-Filter verstärkt Kanten und Texturen im Bild massiv. Dies führt bei Katzen ohne Beute (\\texttt{two\\_cats.mp4}) zu einer leichten Überempfindlichkeit gegenüber Fellstrukturen (Anstieg der Rate von 1,00 auf 3,01 Zustandswechsel). 
Der wahre Wert des Filters offenbart sich jedoch in dunklen Nachtszenen mit echter Beute (\\texttt{black.mp4}): Ohne CLAHE ist das Modell quasi "blind" und verliert die Beute im Dunkeln permanent (Rate schießt auf 8,75 Wechsel pro 100 Frames in die Höhe). Mit CLAHE wird die Rate auf extrem stabile 0,42 Wechsel reduziert. Der Kontrastfilter ist somit für den Nachteinsatz essenziell und unabdingbar, während seine Nebenwirkungen bei Tag vom Zustandsautomaten geglättet werden.

""",
    text,
    flags=re.DOTALL
)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated 6.7 and 6.8 text successfully!')
