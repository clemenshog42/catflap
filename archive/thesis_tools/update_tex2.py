import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r"\\label\{fig:ablation_states\}(.*?)\\subsection\{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens\}"

replacement = r"""\label{fig:ablation_states}
\end{figure}

Die Ergebnisse (siehe Abbildung \ref{fig:ablation_states}) zeigen drastische Unterschiede. Da die Testvideos unterschiedlich lang sind, wurde die Metrik zur besseren Vergleichbarkeit auf \glqq Zustandswechsel pro 100 Frames\grqq\ normalisiert. Besonders bei komplexen Szenarien (wie im Video \texttt{two\_cats.mp4}, in dem sich zwei Katzen im Bildausschnitt befinden) führt das Abschalten der zeitlichen Glättung zu einem extrem unruhigen System. Ohne den Zustandsautomaten lag die Rate bei über 10 Wechseln pro 100 Frames -- die Klappe flackerte extrem unruhig auf und zu, da das Modell bei einzelnen ungünstigen Bildern (Motion Blur, Schatten) kurzzeitig fälschlicherweise eine Beute erkannte. 

Durch das Hinzufügen der \texttt{history\_length=15} wurde das System massiv beruhigt und die Fehlerate radikal reduziert. 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{images/ablation_state_changes_with_prey.png}
    \caption{Auswirkung des Zustandsautomaten auf die Systemstabilität bei Videos \emph{mit} Beute (True Positives).}
    \label{fig:ablation_states_prey}
\end{figure}

Gleichzeitig zeigt die Ablationsstudie bei den Videos \emph{mit Beute} (siehe Abbildung \ref{fig:ablation_states_prey}), dass die zeitliche Glättung auch hier enorm wichtig für die Hardware-Stabilität ist. Während das reine Modell (ohne Glättung) durch kurzzeitige Verdeckungen im Bild (z.\,B. Pfote vor dem Maul) flackerte und oft mehr als 4 Wechsel pro 100 Frames aufwies, sorgte der Zustandsautomat dafür, dass das System durchgängig geglättet wurde. In allen Videos mit echter Beute gab es dank des Automaten exakt einen einzigen, perfekten Zustandswechsel: Das sofortige, dauerhafte Verriegeln der Klappe. Dies belegt eindrucksvoll, dass die Implementierung der \texttt{StateMachine} essenziell ist, um die Systemstabilität im realen Betrieb zu gewährleisten.

\subsection{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens}"""

new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

if new_text == text:
    print("Could not find the target text!")
else:
    with io.open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("Updated analysen.tex successfully!")
