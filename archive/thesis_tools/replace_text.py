import io
import re

file_path = 'C:/Public/Studium/Bachelorarbeit/thesis/analysen.tex'
with io.open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"Die Ergebnisse der Evaluation sind äußerst aufschlussreich und validieren die getroffenen Design-Entscheidungen, wenngleich sie auch die inhärenten Herausforderungen der Bildklassifikation aufzeigen."
replacement = r"""Die Ergebnisse der Evaluation sind äußerst aufschlussreich und validieren die getroffenen Design-Entscheidungen, wenngleich sie auch die inhärenten Herausforderungen der Bildklassifikation aufzeigen.

Diese ereignisbasierte Betrachtungsweise auf Videoebene hat jedoch eine methodische Schwäche: Sie verdeckt die zeitliche Stabilität des Systems. Ein Video, das in der Matrix als korrekt klassifiziertes Ereignis (z.\,B. True Positive) gewertet wird, sagt nichts darüber aus, ob der Zustand während der gesamten Videodauer stabil gehalten wurde oder ob die Zugangskontrolle mehrfach zwischen \glqq verriegelt\grqq\ und \glqq entriegelt\grqq\ schwankte. Ein solches Flackern stellt in der Praxis ein hohes Sicherheitsrisiko dar. Aus diesem Grund reicht eine reine Wahrheitsmatrix zur Evaluierung von Videostreams nicht aus. In den folgenden Ablationsstudien wird daher eine feingranulare zeitliche Metrik (Zustandswechsel pro 100 Frames) herangezogen, um die wahre Robustheit des Systems zu bewerten."""

new_text = text.replace(target, replacement)

if new_text != text:
    with io.open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Erfolgreich ersetzt!')
else:
    print('Fehler: Text nicht gefunden.')
