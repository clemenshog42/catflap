import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Update 6.6
text = re.sub(
    r"Ohne den Zustandsautomaten lag die Rate bei 4,01 Wechseln pro 100 Frames.*?(?=Durch das Hinzufügen)",
    r"Ohne den Zustandsautomaten lag die Rate bei 2,93 Wechseln pro 100 Frames – die Klappe flackerte extrem unruhig auf und zu, da das Modell bei einzelnen ungünstigen Bildern (Motion Blur, Schatten) kurzzeitig fälschlicherweise eine Beute erkannte. \n\n",
    text,
    flags=re.DOTALL
)

text = re.sub(
    r"Durch das Hinzufügen der \\texttt\{history_length=15\} wurde das System massiv beruhigt und die Rate auf nahezu konstante 0,43 Wechsel pro 100 Frames \(ein einziger kurzer Ausreißer im gesamten langen Video\) reduziert\.",
    r"Durch das Hinzufügen der \\texttt{history_length=15} und der überarbeiteten Warte-Logik wurde das System massiv beruhigt und die Rate auf nahezu perfekte 0,21 Wechsel pro 100 Frames reduziert.",
    text
)

# Update 6.7
text = re.sub(
    r"Wie in Abbildung \\ref\{fig:ablation_crop\} zu sehen ist, führt das symmetrische Zuschneiden bei den Testvideos mit potenziellen Fehlalarmen \(Katzen ohne Beute\) zu einer geringfügig höheren Instabilität\. Im längsten Video \(\\texttt\{two_cats\.mp4\}\) stieg die Fehlerquote von 0,43 auf 0,57 Zustandswechsel pro 100 Frames an\. Dies lässt sich dadurch erklären, dass der Klassifikator beim symmetrischen Zuschneiden häufiger Ränder des Halsbandes oder von Schattenwürfen fokussiert, die durch das symmetrische Raster dominanter im Bild liegen\.",
    r"Wie in Abbildung \\ref{fig:ablation_crop} zu sehen ist, führt das symmetrische Zuschneiden auf dem vorliegenden Testdatensatz zu keiner messbaren Verschlechterung mehr. Beide Varianten weisen eine perfekte Stabilität ohne unnötige Zustandswechsel auf. Dies ist ein direktes Resultat der stark verbesserten Robustheit des Zustandsautomaten, welcher vereinzelte Fehlklassifikationen durch Symmetrie-Artefakte (z.\,B. Halsbänder oder Schatten) nun vollständig herausfiltert.",
    text
)

text = re.sub(
    r"Interessanterweise zeigten die Videos \\emph\{mit Beute\} bei beiden Methoden durchgehend eine 100\,\%ige Trefferquote \(jeweils exakt 1 Zustandswechsel zur Verriegelung\)\. Dies deutet darauf hin, dass das asymmetrische Zuschneiden zwar nicht zwingend erforderlich ist, um große Beutetiere zu erkennen, aber dennoch als stabilisierender Faktor gegen Fehlalarme wirkt, da es den Fokusbereich des Klassifikators optimaler auf den zu erwartenden Bereich herabhängender Beute zentriert\.",
    r"Auch bei den Videos \\emph{mit Beute} zeigten beide Methoden durchgehend eine perfekte Erkennung (jeweils exakt 1 Zustandswechsel zur Verriegelung). Dies deutet darauf hin, dass das asymmetrische Zuschneiden zwar nicht zwingend erforderlich ist, um Beutetiere zu erkennen, es jedoch theoretisch den Fokusbereich des Klassifikators optimaler auf den zu erwartenden Bereich herabhängender Beute zentriert und daher als Best Practice beibehalten wird.",
    text
)


with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated 6.6 and 6.7 text successfully!')
