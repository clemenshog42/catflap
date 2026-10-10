import io

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

split1 = text.split(r'\label{fig:ablation_states}')
if len(split1) != 2:
    print('Failed at split1')
    exit(1)

split2 = split1[1].split(r'\subsection{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens}')
if len(split2) != 2:
    print('Failed at split2')
    exit(1)

replacement = r"""
\end{figure}

Die Ergebnisse (siehe Abbildung \ref{fig:ablation_states}) zeigen drastische Unterschiede. Da die Testvideos unterschiedlich lang sind, wurde die Metrik zur besseren Vergleichbarkeit auf \glqq Zustandswechsel pro 100 Frames\grqq\ normalisiert. Besonders bei komplexen Szenarien (wie im Video \texttt{two\_cats.mp4}, in dem sich zwei Katzen im Bildausschnitt befinden) fhrt das Abschalten der zeitlichen Glttung zu einem extrem unruhigen System. Ohne den Zustandsautomaten lag die Rate bei ber 10 Wechseln pro 100 Frames -- die Klappe flackerte extrem unruhig auf und zu, da das Modell bei einzelnen ungnstigen Bildern (Motion Blur, Schatten) kurzzeitig flschlicherweise eine Beute erkannte. 

Durch das Hinzufgen der \texttt{history\_length=15} wurde das System massiv beruhigt und die Fehlerate radikal reduziert. 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{images/ablation_state_changes_with_prey.png}
    \caption{Auswirkung des Zustandsautomaten auf die Systemstabilitt bei Videos \emph{mit} Beute (True Positives).}
    \label{fig:ablation_states_prey}
\end{figure}

Gleichzeitig zeigt die Ablationsstudie bei den Videos \emph{mit Beute} (siehe Abbildung \ref{fig:ablation_states_prey}), dass die zeitliche Glttung auch hier enorm wichtig fr die Hardware-Stabilitt ist. Whrend das reine Modell (ohne Glttung) durch kurzzeitige Verdeckungen im Bild (z.\,B. Pfote vor dem Maul) flackerte und oft mehr als 4 Wechsel pro 100 Frames aufwies, sorgte der Zustandsautomat dafr, dass das System durchgngig geglttet wurde. In allen Videos mit echter Beute gab es dank des Automaten exakt einen einzigen, perfekten Zustandswechsel: Das sofortige, dauerhafte Verriegeln der Klappe. Dies belegt eindrucksvoll, dass die Implementierung der \texttt{StateMachine} essenziell ist, um die Systemstabilitt im realen Betrieb zu gewhrleisten.

"""

new_text = split1[0] + r'\label{fig:ablation_states}' + replacement + r'\subsection{Ablationsstudie: Relevanz des asymmetrischen Zuschneidens}' + split2[1]

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Updated successfully using split!')
