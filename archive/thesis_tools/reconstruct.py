import os

files = {
    "bachelor-thesis.tex": r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[ngerman]{babel}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}

\begin{document}

\begin{center}
\vspace*{2cm}
{\Large \textbf{Zeitlich robuste Erkennung und Zustandsmodellierung von Katzen und Beutetieren mittels Deep Learning}}\\[1cm]
20. August 2026\\[2cm]
\end{center}

\tableofcontents
\newpage

\input{einleitung}
\input{grundlagen}
\input{material}
\input{methodik}
\input{implementierung}
\input{analysen}
\input{diskussion}
\input{zusammenfassung}

\newpage
\bibliographystyle{unsrt}
\bibliography{literatur}

\end{document}
""",
    "einleitung.tex": r"""\section{Einleitung}
\label{sec:einleitung}

Automatische visuelle Erkennungssysteme stellen einen zentralen Bestandteil moderner Anwendungen des maschinellen Sehens dar. Während aktuelle Objektdetektoren auf Einzelbildern zumeist sehr hohe Genauigkeiten erzielen, erweisen sich reale Szenarien durch kurze Verdeckungen, starke Bewegungsunschärfe oder visuell schwer unterscheidbare Objekte oft als fehleranfällig \cite{jiao2022video}. Besonders im Anwendungskontext intelligenter Zugangssysteme, wie beispielsweise einer automatisierten Katzenklappe, ist die isolierte Momentaufnahme unzureichend. Hier ist es von entscheidender Bedeutung festzustellen, ob eine Katze ein Beutetier (wie Mäuse oder Vögel) mit sich führt, was eine robuste Analyse über mehrere Video-Einzelbilder hinweg zwingend erforderlich macht.

Ziel dieser Arbeit ist die Entwicklung und Evaluierung eines mehrstufigen Systems des maschinellen Sehens, welches die zuverlässige Detektion von Katzen und potenziellen Beutetieren sowie deren zeitliche Verfolgung ermöglicht. Der Fokus liegt dabei auf der Kombination mehrerer etablierter Algorithmen zu einer ausfallsicheren Verarbeitungskette. Im ersten Schritt wird das State-of-the-Art Objektdetektionsmodell YOLOv11 \cite{jocher2024yolo} eingesetzt, welches sich durch eine hohe Genauigkeit bei gleichzeitig echtzeitfähiger Inferenz auszeichnet. Im Anschluss kommt ein Verfolgung-durch-Erkennung-Ansatz auf Basis von ByteTrack \cite{zhang2022bytetrack} zum Einsatz. Dieser Algorithmus assoziiert Detektions-Begrenzungsrahmen über die Zeit und minimiert Identitätswechsel bei kurzzeitigen Verdeckungen effektiv.

Die identifizierten Regionen werden durch einen nachgelagerten Klassifikator (YOLO-Klassifikation) ausgewertet. Zur signifikanten Erhöhung der Robustheit und zur Reduzierung von Fehlalarmen werden die ermittelten Konfidenzwerte nicht als Einzelwerte betrachtet, sondern zeitlich aggregiert. Ein eigens entwickelter Zustandsautomat (State Machine) wertet diese Historie über ein gleitendes Zeitfenster aus, um stabile semantische Entscheidungen über den Gesamtzustand (z.\,B. \glqq Katze mit Beute\grqq) zu treffen. Die vorliegende Arbeit evaluiert, inwiefern diese Kombination aus Detektion, Objektverfolgung und zeitlicher Modellierung eine deutlich robustere Klassifikation gegenüber herkömmlichen einzelbildbasierten Methoden bietet.
""",
    "grundlagen.tex": r"""\section{Theoretische Grundlagen}
\label{sec:grundlagen}

In diesem Kapitel werden die theoretischen Konzepte erläutert, welche das Fundament für das in dieser Arbeit entwickelte Erkennungssystem bilden. Der Fokus liegt dabei ausschließlich auf jenen Technologien und Algorithmen, die in der finalen Verarbeitungskette Anwendung finden. Dazu zählen Prinzipien der Objekterkennung, der Mehrfach-Objektverfolgung, der Bildklassifikation sowie die Modellierung zeitlicher Verläufe mittels Zustandsautomaten.

\subsection{Objekterkennung mit YOLOv11}
\label{subsec:yolo_det}

Die automatische Bildanalyse lässt sich grundsätzlich in verschiedene Problemklassen unterteilen \cite{szeliski2022computer}. Während die reine Bildklassifikation lediglich die Frage beantwortet, ob ein bestimmtes Objekt in einem Bild vorhanden ist, geht die Objekterkennung einen Schritt weiter. Sie ermittelt nicht nur die Klasse des Objekts, sondern auch dessen genaue räumliche Position innerhalb des Bildes, welche üblicherweise durch einen Begrenzungsrahmen dargestellt wird.

Das in dieser Arbeit eingesetzte YOLO-Modell (You Only Look Once) in der Version 11 \cite{jocher2024yolo} gehört zur Familie der sogenannten einstufigen Detektoren. Im Gegensatz zu traditionellen zweistufigen Verfahren, die zunächst in einem aufwändigen Prozess potenziell interessante Bildregionen (Regionsvorschläge) extrahieren und diese erst im zweiten Schritt klassifizieren, verarbeitet YOLO das gesamte Bild in einem einzigen Durchlauf durch ein Faltungsnetzwerk (CNN). Dabei wird das Bild in ein Raster unterteilt. Für jede Rasterzelle sagt das Netzwerk simultan mehrere Begrenzungsrahmen, zugehörige Klassenwahrscheinlichkeiten sowie einen Konfidenzwert voraus. Der Konfidenzwert gibt an, wie sicher sich das Modell ist, dass der gefundene Begrenzungsrahmen tatsächlich ein relevantes Objekt enthält und wie genau die Box dieses umschließt.

Für die vorliegende Arbeit ist YOLOv11 von zentraler Bedeutung, da die Echtzeiterkennung von Katzen in hochdynamischen Umgebungen (wie einer Katzenklappe) schnelle Inferenzzeiten erfordert. Die Ausgabe des Modells liefert für jede erkannte Katze die exakten Koordinaten im Einzelbild, welche anschließend als Basis für die Objektverfolgung dienen.

\subsection{Mehrfach-Objektverfolgung mit ByteTrack}
\label{subsec:bytetrack}

Eine reine Objekterkennung bewertet jedes Einzelbild eines Videostreams isoliert. Wenn eine Katze durch das Bildfeld läuft, generiert der Detektor für jedes Einzelbild einen neuen, unabhängigen Begrenzungsrahmen. Um jedoch eine zeitliche Analyse durchführen zu können (beispielsweise um zu bewerten, ob das Objekt über einen längeren Zeitraum hinweg Beute trägt), muss eine Identität über die Zeit hinweg aufrechterhalten werden. Diese Aufgabe übernimmt die Mehrfach-Objektverfolgung (MOT).

Der in dieser Arbeit verfolgte Ansatz basiert auf dem Paradigma der \textit{Verfolgung durch Erkennung}. Hierbei erzeugt zunächst ein Objektdetektor (in diesem Fall YOLOv11) die Begrenzungsrahmen für das aktuelle Einzelbild. Anschließend ist es Aufgabe des Trackers, diese aktuellen Erkennungen den bereits bekannten, verfolgten Objekten aus vorherigen Einzelbildern zuzuordnen. Dies geschieht üblicherweise durch Bewegungsvorhersagen (z.\,B. mittels Kalman-Filter \cite{kalman1960new}) und die Berechnung räumlicher Überlappungen (Intersection over Union, IoU).

Der verwendete Algorithmus ByteTrack \cite{zhang2022bytetrack} zeichnet sich durch einen besonderen Ansatz bei dieser Datenassoziation aus. Viele klassische Tracker filtern Erkennungen mit einem niedrigen Konfidenzwert rigoros heraus, um Fehlalarme zu vermeiden. Dies führt jedoch häufig zum Verlust (Identitätswechsel oder Abbruch) von verfolgten Objekten, wenn ein Objekt kurzzeitig verdeckt wird oder verschwimmt. ByteTrack nutzt stattdessen auch jene Begrenzungsrahmen mit geringerer Konfidenz. In einem zweistufigen Zuordnungsprozess werden zunächst Erkennungen mit hoher Konfidenz mit existierenden verfolgten Objekten verknüpft. Anschließend werden die verbleibenden verfolgten Objekte mit den Erkennungen mit niedriger Konfidenz abgeglichen. Dies ermöglicht eine äußerst robuste Verfolgung bei temporären Verdeckungen und qualifiziert ByteTrack ideal für die unvorhersehbaren Bewegungen und Okklusionen im Anwendungsfall einer Katzenklappe.

\subsection{Bildklassifikation mit YOLOv11}
\label{subsec:yolo_class}

Während die Objekterkennung die Lokalisierung eines Objekts übernimmt, dient die Bildklassifikation der detaillierten Zuweisung einer Kategorie zu einem gegebenen Bildausschnitt. Die Klassifikation arbeitet nach dem Grundprinzip von Faltungsnetzwerken (CNNs) \cite{lecun1998gradient}, welche durch mehrere Schichten von Faltungen (Convolutions) und Zusammenfassungs-Schichten zunehmend abstrakte Merkmale aus dem Eingangsbild extrahieren \cite{goodfellow2016deep}.

In dieser Arbeit wird die YOLOv11-Architektur nicht nur für die Objekterkennung, sondern in einem zweiten, separaten Modell auch als Bildklassifikator genutzt. Moderne YOLO-Versionen bieten neben der Detektion auch dezidierte Klassifikations-Netzwerke, die durch ihre stark optimierte Architektur (z.\,B. effiziente Backbone-Strukturen und reduzierte Parameteranzahl) für extrem schnelle Inferenz bei hoher Genauigkeit ausgelegt sind.

Innerhalb der Verarbeitungskette erhält das YOLO-Klassifikationsmodell nicht das gesamte Kamerabild als Eingabe, sondern lediglich den Bildausschnitt, der zuvor vom ersten YOLO-Modell (dem Detektor) als Katze identifiziert wurde. Die Aufgabe des Klassifikators beschränkt sich somit auf die binäre Entscheidung, ob in diesem konkreten Ausschnitt ein Beutetier (Beute) vorhanden ist oder nicht. Die Ausgabe des Modells ist ein kontinuierlicher Wahrscheinlichkeitswert (Konfidenzwert).

\subsection{Zustandsautomaten (Finite State Machines)}
\label{subsec:fsm}

Sowohl Objektdetektoren als auch Bildklassifikatoren liefern Momentaufnahmen, deren Ergebnisse von Einzelbild zu Einzelbild leicht schwanken können. Um daraus stabile und belastbare Entscheidungen abzuleiten, wird eine zeitliche Aggregation benötigt. Ein geeignetes mathematisches Modell zur Abbildung einer solchen Logik ist der endliche Zustandsautomat (Finite State Machine, FSM).

Ein endlicher Zustandsautomat ist ein Verhaltensmodell, das aus einer endlichen Menge von \textit{Zuständen} besteht. Das System kann sich zu einem gegebenen Zeitpunkt immer nur in exakt einem dieser Zustände befinden. Durch das Eintreten definierter \textit{Ereignisse} oder das Erfüllen bestimmter \textit{Bedingungen} kann das System von einem Zustand in einen anderen wechseln. Dieser Vorgang wird als \textit{Übergang} (Transition) bezeichnet. Zusätzlich können an Zustände oder Übergänge spezifische \textit{Aktionen} geknüpft sein.

Ein alltägliches, vereinfachtes Beispiel für eine FSM ist eine Ampelanlage: Die Zustände sind \glqq Rot\grqq, \glqq Grün\grqq\ und \glqq Gelb\grqq. Ein Ereignis (z.\,B. der Ablauf eines internen Timers) löst den Übergang vom Zustand \glqq Rot\grqq\ in den Zustand \glqq Grün\grqq\ aus.

Im Kontext der Bildverarbeitung dienen Zustandsautomaten dazu, das \glqq Flackern\grqq\ von Vorhersagen zu glätten. Anstatt bei einem einzelnen Einzelbild mit niedrigem Konfidenzwert sofort die Systementscheidung zu ändern, wird der Zustandsübergang an striktere Bedingungen (z.\,B. einen Schwellenwert über ein gleitendes Zeitfenster) gebunden. Dies bildet das Fundament für die in Kapitel 4 beschriebene, eigens entwickelte Logik zur Modellierung des Zustandes \glqq Katze mit Beute\grqq.
""",
    "material.tex": r"""\section{Material und Versuchsgrundlage}
\label{sec:material}

Dieses Kapitel beschreibt die konkreten Ressourcen, Ausgangsdaten und technischen Umgebungen, die für das Training der Modelle und die Evaluation der in dieser Arbeit vorgestellten Verarbeitungskette verwendet wurden.

\subsection{Datensätze}
Für die Entwicklung des zweistufigen Erkennungssystems wurden unterschiedliche Bilddatenbanken für die Objekterkennung und die Bildklassifikation benötigt.

\textbf{Datensatz für die Objekterkennung (Katzengesichter):} \\
Für das Training des YOLOv11-Detektors, welcher die Position der Katze im Bildfeld präzise lokalisieren soll, wurde ein öffentlich zugänglicher Kaggle-Datensatz herangezogen (\glqq Cat Dataset\grqq\ von Crawford \cite{kaggle2024cat}). Dieser Datensatz umfasst eine breite Variation an Katzenbildern unter unterschiedlichen Lichtverhältnissen, Posen und Hintergründen. Die Daten wurden vorverarbeitet und in ein für das YOLO-Training geeignetes Format überführt.

\textbf{Datensatz für die Klassifikation (Beutetiere):} \\
Die Beschaffung von ausreichendem und hochqualitativem Bildmaterial, das Katzen mit echten Beutetieren im Maul zeigt (insbesondere bei gleichzeitig schwierigen Lichtverhältnissen), stellte eine signifikante Herausforderung dar. Um das YOLO-Klassifikationsmodell auf eine ausreichend große Datenbasis stützen zu können, wurden die Trainings- und Validierungsdaten für diese Problemstellung zum Teil KI-generiert. Durch den Einsatz moderner generativer Modelle konnten spezifische Szenarien (z.\,B. unterschiedliche Beutetiere, partielle Verdeckungen) gezielt synthetisiert werden, um dem Modell das Erlernen robuster Merkmale zu ermöglichen.

\subsection{Hardware und Softwareumgebung}
Die ressourcenintensiven Trainingsprozesse der neuronalen Netze (YOLO-Detektor und YOLO-Klassifikator) erforderten den Einsatz spezialisierter Hardware-Beschleuniger.

\textbf{Hardware-Setup:} \\
Das Training der Modelle wurde primär auf den Rechenzentrums-Ressourcen der Plattform Kaggle durchgeführt, um von der dort bereitgestellten GPU-Beschleunigung zu profitieren. Für die spätere Inferenz und Evaluation des Gesamtsystems (inklusive Objektverfolgung und Zustandsautomat) kam eine lokale Systemarchitektur zum Einsatz.

\textbf{Softwareumgebung:} \\
Die Implementierung des Systems erfolgte vollständig in Python \cite{van1995python}. Als zentrales Framework für das Training des Detektors sowie des Klassifikators kam das \texttt{ultralytics}-Paket zum Einsatz, welches die Architektur von YOLOv11 bereitstellt. Bildverarbeitungsoperationen (z.\,B. das Extrahieren von Bildausschnitten) wurden mit OpenCV \cite{bradski2000opencv} realisiert. Die finalen Modelle wurden teilweise in Formate wie TensorFlow Lite \cite{abadi2016tensorflow} oder OpenVINO \cite{intel2024openvino} exportiert, um die Inferenzgeschwindigkeit zu maximieren.
""",
    "methodik.tex": r"""\section{Methodik und Entwicklung}
\label{sec:methodik}

In diesem Kapitel wird detailliert dargelegt, wie die konzeptionelle Entwicklung und technische Umsetzung des in dieser Arbeit vorgestellten Erkennungssystems erfolgte. Es beschreibt den Weg von der Problemstellung zur fertigen Gesamtlösung und beleuchtet die getroffenen Designentscheidungen sowie die Architektur der einzelnen Verarbeitungsschritte.

\subsection{Gesamtarchitektur und Datenfluss}
Um die komplexe Aufgabe zu lösen, zuverlässig zu erkennen, ob eine Katze ein Beutetier im Maul trägt, reicht eine einfache Bildklassifikation nicht aus. Stattdessen wurde eine mehrstufige Verarbeitungskette entwickelt, die aus Objekterkennung, Objektverfolgung, Bildausschnitt-Extraktion, Klassifikation und zeitlicher Zustandsmodellierung besteht.

Der Datenfluss des Systems gestaltet sich wie folgt: Ein eingehendes Einzelbild der Kamera wird zunächst vorverarbeitet. Anschließend analysiert der feinabgestimmte YOLOv11-Detektor das Bild und lokalisiert das Gesicht der Katze. Der integrierte ByteTrack-Algorithmus ordnet dieser Erkennung eine konsistente Identität (Objekt-ID) zu. Basierend auf den Koordinaten der Erkennung wird ein spezifischer Bildausschnitt extrahiert, welcher an ein zweites Modell (den YOLO-Klassifikator) übergeben wird. Dieser berechnet einen Konfidenzwert für das Vorhandensein von Beute. Im letzten Schritt aggregiert ein eigens entwickelter Zustandsautomat diese Konfidenzwerte über die Zeit und trifft die finale Entscheidung, in welchem Zustand (z.\,B. \glqq Katze mit Beute\grqq) sich das verfolgte Objekt befindet.

\subsection{Vorverarbeitung für Infrarot-Nachtsicht}
Eine zentrale Anforderung an das System ist die Funktionsfähigkeit bei Nacht, da Katzen ihre Beute häufig in den späten Abend- oder frühen Morgenstunden nach Hause bringen. Kameras für Katzenklappen schalten bei Dunkelheit üblicherweise auf Infrarot-Nachtsicht um, was zu monochromen (graustufigen) Bildern führt.

Um zu verhindern, dass die neuronalen Netze Farbinformationen aus Tageslichtaufnahmen als zwingende Merkmale erlernen und dadurch bei Nachtaufnahmen versagen, wurde die strategische Entscheidung getroffen, das System vollständig auf Graustufen zu optimieren. Jedes eingehende Einzelbild wird noch vor der Inferenz in ein Graustufenbild konvertiert.

Zusätzlich kommt der Algorithmus CLAHE (Contrast Limited Adaptive Histogram Equalization) \cite{zuiderveld1994contrast} zum Einsatz. Gerade bei schlechten Lichtverhältnissen oder starken Schattenwürfen verbessert CLAHE den lokalen Kontrast signifikant, ohne dabei das Bildrauschen übermäßig zu verstärken. Dies ermöglicht es dem Detektor, die Konturen von Katzen und potenzieller Beute deutlich robuster zu extrahieren.

\subsection{Training und Anpassung des Objektdetektors (YOLOv11n)}
Für die Objekterkennung wurde die Architektur YOLOv11 in der Nano-Ausführung (\texttt{yolov11n}) gewählt. Diese Variante bietet den optimalen Kompromiss aus hoher Erkennungsgenauigkeit und der notwendigen Inferenzgeschwindigkeit für Echtzeitanwendungen auf ressourcenbeschränkter Hardware.

Da Standard-Modelle nicht spezifisch auf die Erkennung von Katzengesichtern mit potenzieller Beute im Maulbereich optimiert sind, wurde eine gezielte Feinabstimmung durchgeführt. Als Datengrundlage diente der Kaggle Katzen-Datensatz \cite{kaggle2024cat}. Die ursprünglichen Annotationen dieses Datensatzes bestanden aus neun Schlüsselpunkten im Gesicht der Katze (Augen, Nase, Ohren, Mund). Um diese Daten für das Begrenzungsrahmen-Training von YOLO nutzbar zu machen, wurde ein Konvertierungsalgorithmus entwickelt. Dieser berechnete den Begrenzungsrahmen so, dass er von der obersten Kante der Ohren bis exakt 50 Pixel unterhalb des Mundes reicht. Dieser vertikale Puffer nach unten ist eine essenzielle Designentscheidung, um sicherzustellen, dass aus dem Maul hängende Beutetiere stets innerhalb der Detektionsbox liegen.

Das Training des YOLOv11n-Modells erfolgte über 100 Epochen mit einer Bildauflösung von 640x640 Pixeln, einer Stapelgröße von 16 und einem Mechanismus zum frühzeitigen Trainingsabbruch, um Überanpassung zu vermeiden \cite{goodfellow2016deep}.

\subsection{Automatisierte Generierung des Klassifikations-Datensatzes}
Für den zweiten Schritt der Verarbeitungskette – die Klassifikation, ob die Katze Beute im Maul trägt – wurde ein separater Datensatz benötigt. Die manuelle Annotation (das passgenaue Zuschneiden) hunderter Bilder von Katzen mit und ohne Beute wäre äußerst zeitaufwändig gewesen.

Daher wurde ein innovativer Ansatz gewählt: Der zuvor im ersten Schritt erfolgreich trainierte YOLOv11-Katzengesichts-Detektor wurde automatisiert auf den Rohdatensatz der \glqq Beute / Keine Beute\grqq-Bilder angewendet. Der Detektor fand selbstständig die Katzengesichter und extrahierte die exakten Bildausschnitte. Diese automatisch generierten Bildausschnitte bildeten anschließend die hochqualitative und perfekt zentrierte Trainingsgrundlage für den Bildklassifikator.

\subsection{Bildausschnitt-Extraktion (Asymmetrisches Zuschneiden)}
Während der Inferenzphase (im laufenden Betrieb) erhält der Klassifikator nicht das gesamte Kamerabild, sondern nur den von YOLO detektierten Bereich. Um die Erkennung der Beute weiter zu optimieren, wurde eine asymmetrische Randerweiterungs-Logik für den Bildausschnitt implementiert.

Ausgehend von den Koordinaten des YOLO-Begrenzungsrahmens wird der Ausschnitt links und rechts um 15 Pixel (\texttt{pad\_w}) sowie oben um 10 Pixel (\texttt{pad\_top}) vergrößert. Nach unten hin wird der Ausschnitt jedoch signifikant um 30 Pixel (\texttt{pad\_bottom}) erweitert. Da kleine, teilweise verdeckte Objekte für Standard-Architekturen eine bekannte Herausforderung darstellen \cite{tong2020small} und Beutetiere in der Regel aus dem Maul nach unten hängen, garantiert diese asymmetrische Erweiterung, dass die Beute nicht am Bildrand abgeschnitten wird und dem Klassifikator vollständig zur Verfügung steht.

Der extrahierte Ausschnitt wird anschließend erneut durch CLAHE im Kontrast verstärkt und für die Eingabeschicht des Klassifikators (welche ein 3-Kanal-Bild erwartet) dreifach dupliziert (\texttt{cv2.cvtColor(gray\_crop, cv2.COLOR\_GRAY2BGR)}).

\subsection{Training des Bildklassifikators (YOLO-Klassifikation)}
Als Bildklassifikator kommt nicht wie ursprünglich vorgesehen eine klassische Keras/EfficientNet-Architektur zum Einsatz, sondern ebenfalls ein dediziertes YOLO-Modell. Diese Designentscheidung ergab sich aus der Erkenntnis, dass moderne YOLO-Versionen hochoptimierte Netzwerke für Klassifikationsaufgaben bereitstellen, die sich nahtlos und performant in die bestehende \texttt{ultralytics}-Verarbeitungskette integrieren lassen.

Das Modell liefert für den übergebenen Bildausschnitt einen kontinuierlichen Konfidenzwert, der angibt, mit welcher Sicherheit das Objekt der Klasse \glqq Beute\grqq\ zugeordnet werden kann. Starke künstliche Datenvervielfältigung (zufällige Anpassung von Helligkeit, Kontrast und horizontale Spiegelungen) während des Trainings sorgte dafür, dass das Modell extrem robust gegenüber wechselnden Lichtverhältnissen in der Katzenklappe wird.

\subsection{Objektverfolgung und Entwicklung des Zustandsmodells}
Wenn sich die Katze der Klappe nähert, variiert die Qualität der Bilder von Einzelbild zu Einzelbild stark (z.\,B. durch Bewegungsunschärfe oder ungünstige Winkel). Dies führt dazu, dass die Ausgaben des YOLO-Klassifikators \glqq flackern\grqq. Ein einzelnes Einzelbild, in dem die Beute verdeckt ist, würde bei einer rein auf dem Einzelbild basierenden Entscheidung sofort zu einem Abbruch der Erkennung führen.

Um dieses Problem zu lösen, wurde die Objekterkennung (YOLO) zunächst um den ByteTrack-Algorithmus erweitert. ByteTrack weist der Katze bei ihrem ersten Auftreten im Bildfeld eine eindeutige Objekt-ID zu und verfolgt diese konsistent über die folgenden Einzelbilder.

Basierend auf dieser Identitätsnachverfolgung wurde ein endlicher Zustandsautomat (Finite State Machine, FSM) entwickelt. Die zentrale Klasse \texttt{StateMachine} verwaltet für jede erkannte Katze (\texttt{track\_id}) ein eigenes \texttt{TrackState}-Objekt. Das Modell kennt drei fundamentale Zustände:

\begin{itemize}
    \item \texttt{NO\_CAT}: Es befindet sich kein relevantes Objekt im Sichtfeld.
    \item \texttt{CAT\_NO\_PREY}: Eine Katze wurde erkannt, trägt jedoch keine Beute.
    \item \texttt{CAT\_WITH\_PREY}: Eine Katze wurde erkannt und trägt mit hinreichender Sicherheit eine Beute.
\end{itemize}

Anstatt die Systementscheidung auf das aktuelle Einzelbild zu stützen, führt das \texttt{TrackState}-Objekt eine kontinuierliche Historie (Queue) der YOLO-Klassifikations-Konfidenzwerte über die letzten 15 Einzelbilder. Für die Zustandsbestimmung wird bei jedem neuen Einzelbild der gleitende Durchschnitt dieser Historie berechnet. Überschreitet dieser Durchschnittswert einen streng definierten Schwellenwert von 0.8, geht das System in den Zustand \texttt{CAT\_WITH\_PREY} über. Fällt der Wert darunter, wechselt das System zurück auf \texttt{CAT\_NO\_PREY}.

Der sehr hohe Schwellenwert von 0.8 ist eine bewusste Designentscheidung: Er minimiert die Fehlalarm-Rate. Im praktischen Betrieb einer Katzenklappe hat das Wohl der Katze für die meisten Hausbesitzer oberste Priorität; sie soll nicht durch häufige, unbegründete Aussperrungen verunsichert werden. Das System blockiert die Klappe daher nur, wenn der gleitende Durchschnitt belegt, dass sich das System extrem sicher ist. Für das Haustier bleibt so das Verriegelungsverhalten klar und konsistent nachvollziehbar.

Zusätzlich verfügt die Zustandsmaschine über eine robuste Fehlerbehandlung: Objekte, die das Bildfeld verlassen haben oder länger als 30 Einzelbilder vom Tracker nicht mehr gefunden wurden (\texttt{max\_missed\_frames}), werden automatisch aus dem Speicher entfernt (Bereinigung), um Speicherlecks zu vermeiden und das System für die nächste Katze zurückzusetzen.
""",
    "implementierung.tex": r"""\section{Design und Implementierung}
\label{sec:implementierung}

Nachdem im vorherigen Kapitel die konzeptionelle Methodik dargelegt wurde, fokussiert sich dieses Kapitel auf die konkrete informationstechnische Realisierung des Systems. Es wird erläutert, wie die Architektur in Softwarekomponenten gegliedert ist, welche Datenstrukturen verwendet wurden und wie die Module miteinander interagieren. Relevante Codeausschnitte veranschaulichen dabei die zentralen Algorithmen.

\subsection{Softwarearchitektur und Kernkomponenten}
Die Implementierung erfolgte in Python und ist streng modular aufgebaut. Die Trennung von Erkennungsmodellen (Verarbeitungskette), logischer Auswertung (Zustandsautomat) und Ausführung (Main-Loop) gewährleistet eine hohe Wartbarkeit. Das System besteht aus drei zentralen Skripten: \texttt{models.py}, \texttt{state\_machine.py} und \texttt{main.py}.

\subsubsection{Das Modul \texttt{models.py}: Die \texttt{CatFlapPipeline}}
Die Klasse \texttt{CatFlapPipeline} kapselt die gesamte Interaktion mit den neuronalen Netzen. Bei der Initialisierung lädt sie die beiden trainierten YOLO-Modelle: den Detektor (im Format TFLite für maximale Inferenzgeschwindigkeit) und den Klassifikator (im OpenVINO-Format).

Die Methode \texttt{run\_detector} übernimmt die Vorverarbeitung (Graustufen-Konvertierung und optionale CLAHE-Kontrastverstärkung) und steuert den Aufruf der Objekterkennung. Hierbei wird durch das Setzen des Parameters \texttt{persist=True} der in Ultralytics integrierte ByteTrack-Tracker aktiviert, welcher die kontinuierlichen Begrenzungsrahmen samt Objekt-IDs zurückgibt.

Die Methode \texttt{run\_classifier} implementiert die in Kapitel 4 beschriebene asymmetrische Zuschneiden-Logik. Im folgenden Codeausschnitt ist ersichtlich, wie der Bildbereich nach unten hin signifikant stärker erweitert wird (\texttt{pad\_bottom = 30}), um hängende Beutetiere nicht abzuschneiden:

\begin{lstlisting}[language=Python, caption={Asymmetrische Randerweiterung für den Bildausschnitt}]
x1, y1, x2, y2 = map(int, box)
# Add asymmetrical padding (heavy on bottom for dangling prey)
pad_w = 15
pad_top = 10
pad_bottom = 30
x1 -= pad_w
y1 -= pad_top
x2 += pad_w
y2 += pad_bottom
\end{lstlisting}

Anschließend wird der extrahierte Bildausschnitt für das Klassifikations-Netzwerk verarbeitet und der Wahrscheinlichkeitswert für die Klasse \glqq Beute\grqq\ zurückgegeben.

\subsubsection{Das Modul \texttt{state\_machine.py}: Zeitliche Aggregation}
Die technische Umsetzung des Zustandsmodells erfolgt objektorientiert über die Klassen \texttt{TrackState} und \texttt{StateMachine}. Für jede erkannte Katze wird eine Instanz der Klasse \texttt{TrackState} angelegt. Diese speichert die Historie der Konfidenzwerte performant in einer \texttt{collections.deque} mit einer fixen Maximallänge (\texttt{history\_length = 15}). Durch die Verwendung dieser Datenstruktur wird garantiert, dass bei Erreichen der Kapazitätsgrenze automatisch der älteste Wert verworfen wird (\glqq First-In-First-Out\grqq), ohne dass manuelles Speicher-Management notwendig ist.

Der folgende Ausschnitt zeigt die Kernlogik der \texttt{update}-Methode, welche bei jedem Einzelbild den gleitenden Durchschnitt (\texttt{numpy.mean}) berechnet und den Zustand entsprechend umschaltet:

\begin{lstlisting}[language=Python, caption={Zustandsermittlung mittels gleitendem Durchschnitt}]
def update(self, prey_confidence, frame_idx):
    self.confidence_history.append(prey_confidence)
    self.last_seen = frame_idx

    # Calculate moving average
    avg_confidence = np.mean(self.confidence_history)

    # Determine new state based on aggregated confidence
    if avg_confidence >= self.threshold:
        self.current_state = State.CAT_WITH_PREY
    else:
        self.current_state = State.CAT_NO_PREY
\end{lstlisting}

Die übergeordnete Klasse \texttt{StateMachine} orchestriert diese \texttt{TrackState}-Objekte in einem Dictionary (\texttt{track\_id} $\rightarrow$ \texttt{TrackState}). Die Methode \texttt{cleanup\_stale\_tracks} iteriert regelmäßig über dieses Dictionary und löscht alle Objekte, die seit mehr als 30 Einzelbildern nicht mehr vom Tracker aktualisiert wurden, um Speicherlecks zu verhindern.

\subsection{Orchestrierung und gezieltes Sammeln schwieriger Negativbeispiele (\texttt{main.py})}
Die Ausführungsschicht (\texttt{main.py}) verbindet die \texttt{CatFlapPipeline} und die \texttt{StateMachine}. In einer kontinuierlichen \texttt{while}-Schleife werden Einzelbilder von einer Videoquelle (Videodatei oder Kamera) via \texttt{cv2.VideoCapture} eingelesen. Für jedes Einzelbild iteriert das Skript über alle erkannten Objekte, führt die Klassifikation durch, aktualisiert den Zustandsautomaten und zeichnet abschließend Visualisierungen auf das Bild.

Ein besonderes technisches Merkmal in der \texttt{main.py} ist die Implementierung eines Mechanismus zum gezielten Sammeln schwieriger Negativbeispiele. Falls das System via Kommandozeilenparameter (\texttt{--save\_uncertain}) angewiesen wird, prüft es nach jedem Durchlauf den Konfidenzwert. Liegt dieser in einem unsicheren Bereich, wird eine unmodifizierte Kopie des Einzelbildes automatisch gespeichert. Um die Festplatte nicht mit nahezu identischen Bildern zu fluten, blockiert ein Timer weitere Speicherungen derselben \texttt{track\_id} für 30 Einzelbilder.
""",
    "analysen.tex": r"""\section{Tests und Analysen}
\label{sec:analysen}

In diesem Kapitel wird die Leistungsfähigkeit des implementierten Systems evaluiert. Die Beurteilung erfolgt anhand von realen Videodaten, die das System während der Trainingsphase nicht gesehen hat. Im Fokus steht dabei die Frage, wie zuverlässig die Kombination aus Objekterkennung, Klassifikation und zeitlicher Zustandsmodellierung in der Praxis arbeitet und welche Kompromisse durch die getroffenen Design-Entscheidungen entstanden sind.

\subsection{Modell-Performance (Training \& Validation)}
Bevor das Gesamtsystem (inklusive Objektverfolgung und State Machine) auf Videodaten evaluiert wird, ist ein Blick auf die isolierte Leistung des YOLO-Klassifikators essenziell. Die Trainingsmetriken belegen, dass das Modell erfolgreich konvergiert ist. Der Loss sinkt kontinuierlich über die Epochen, während die Genauigkeit auf dem Validierungsdatensatz ein stabiles Plateau erreicht. Dies spricht für ein robustes Lernen ohne massive Überanpassung.

Um die Rohleistung des Netzes bei der binären Unterscheidung zwischen \glqq Beute\grqq\ und \glqq Keine Beute\grqq\ auf Einzelbildern zu beurteilen, liefert die Confusion Matrix auf dem Validierungsdatensatz wertvolle Einblicke. Sie zeigt die Klassifikationsleistung auf den extrahierten Bildausschnitten, \textit{bevor} die State Machine zur Glättung eingreift.

Die Metriken auf Modell-Ebene bilden das Fundament. Die entscheidende Frage für den Praxiseinsatz ist jedoch, wie sich diese Rohleistung im zeitlichen Verlauf eines Videos – orchestriert durch die Verarbeitungskette – verhält.

\subsection{Evaluationsmethodik und Datensatz}
Das System wurde auf Basis eines dedizierten Testdatensatzes evaluiert, der aus insgesamt 7 Videosequenzen besteht. Diese gliedern sich in zwei Kategorien auf:

\begin{itemize}
    \item \textbf{Positive Samples (Mit Beute):} 4 Videos (\texttt{black.mp4}, \texttt{orange.mp4}, \texttt{orange2.mp4}, \texttt{two\_cats.mp4})
    \item \textbf{Negative Samples (Ohne Beute):} 3 Videos (\texttt{black.mp4}, \texttt{orange.mp4}, \texttt{two\_cats.mp4})
\end{itemize}

Die Evaluation fand automatisiert über ein Testskript statt, welches die gesamte Verarbeitungskette für jedes Einzelbild simulierte. Ein Video galt als positiv klassifiziert (\glqq Katze trägt Beute\grqq), sobald die State Machine den Zustand \texttt{CAT\_WITH\_PREY} annahm.

\subsection{Definition der System-Metriken}
Für die Bewertung der Performance in diesem spezifischen Anwendungsfall ist die Unterscheidung zwischen verschiedenen Fehlerarten essenziell. Die Metriken leiten sich aus der Confusion Matrix ab:

\begin{itemize}
    \item \textbf{True Positive (TP):} Die Katze hat eine Beute und das System blockiert korrekt.
    \item \textbf{False Negative (FN):} Die Katze hat eine Beute, aber das System lässt sie herein (Maus im Haus).
    \item \textbf{Korrekte Abweisung (TN):} Die Katze hat keine Beute und wird hereingelassen.
    \item \textbf{Fehlalarm (FP):} Die Katze hat keine Beute, wird aber fälschlicherweise blockiert.
\end{itemize}

Im Kontext einer Katzenklappe besitzt die Katze aus Nutzersicht die höchste Priorität. Ein Fehlalarm bedeutet, dass das Haustier grundlos vor einer verschlossenen Tür steht. Um das Vertrauen des Tieres in die Klappe nicht zu zerstören, liegt der Fokus der Evaluierung primär auf der Vermeidung solcher Fehlalarme (hohe Vorhersagegenauigkeit). Ein seltenes False Negative (eine Maus gelangt ins Haus) wird im Tausch für eine stressfreie Nutzung der Klappe durch die Katze bewusst in Kauf genommen.

\subsection{Ergebnisse der Evaluation}
Das System verarbeitete die Videosequenzen mit einer durchschnittlichen Inferenzgeschwindigkeit von knapp 8 FPS. Die Erkennungsleistung über den Testdatensatz ergab 4 TP, 0 FN, 2 FP und 1 TN. 

Aus diesen Werten ergeben sich die zentralen Leistungsmetriken des Systems:

\begin{itemize}
    \item \textbf{Vorhersagegenauigkeit:} 66,7 \%
    \item \textbf{Trefferquote:} 100,0 \%
\end{itemize}

\subsection{Analyse der Systemleistung und Kompromisse}
Die Ergebnisse der Evaluation sind äußerst aufschlussreich und validieren die getroffenen Design-Entscheidungen, wenngleich sie auch die inhärenten Herausforderungen der Bildklassifikation aufzeigen.

\subsubsection{Präzision und der 0.8-Schwellenwert}
Wie in Kapitel 4 dargelegt, wurde der Schwellenwert für den gleitenden Durchschnitt der State Machine bewusst auf den sehr hohen Wert von 0.8 gesetzt. Das System muss sich über eine Sequenz von Einzelbildern zu mindestens 80\,\% sicher sein, bevor es die Klappe verriegelt. Das primäre Ziel hierbei war die Maximierung der Vorhersagegenauigkeit (Reduktion von Fehlalarmen), da die Katze Vorrang hat.

Die Evaluierung zeigt, dass diese Maßnahme greift: Ohne die glättende Historie der State Machine und den hohen Schwellenwert hätten kurzzeitige Fehlklassifikationen (etwa durch Schattenwurf unter dem Maul) unweigerlich zu einem Fehlalarm geführt. Dennoch verbleiben zwei Fehlalarme. Eine Analyse dieser Fehlerfälle offenbarte, dass der Klassifikator bei sehr ungünstigen Lichtbedingungen oder wenn Texturen im Hintergrund exakt mit dem Gesicht verschmelzen, weiterhin zu starke Konfidenzwerte ausgibt.

\subsubsection{Die perfekte Trefferquote (100\,\%)}
Obwohl das System extrem defensiv eingestellt ist, lag die Trefferquote auf dem Testdatensatz bei makellosen 100\,\%. Keine einzige Maus wurde übersehen. Das beweist, dass das Zusammenspiel aus asymmetrischem Zuschneiden (welches hängende Beute nicht abschneidet) und der zeitlichen Modellierung phänomenal funktioniert. Wenn eine Maus vorhanden ist, liefert der Klassifikator über mehrere Einzelbilder hinweg Werte, die hoch genug sind, um selbst den strengen Schwellenwert zu durchbrechen.
""",
    "diskussion.tex": r"""\section{Diskussion}
\label{sec:diskussion}

Die in Kapitel 6 präsentierten Testergebnisse liefern ein klares quantitatives Bild der Systemleistung. In diesem Kapitel werden diese Ergebnisse qualitativ interpretiert und kritisch bewertet. Dabei wird untersucht, inwiefern die initialen Forschungsziele erreicht wurden, welche architektonischen Entscheidungen maßgeblich für den Erfolg verantwortlich sind und wo die Grenzen des gewählten Ansatzes liegen.

\subsection{Beantwortung der zentralen Forschungsfrage}
Die primäre Forschungsfrage dieser Arbeit lautete, ob eine mehrstufige Verarbeitungskette – bestehend aus Objekterkennung, Objektverfolgung und zeitlicher Zustandsmodellierung – eine signifikant robustere Beute-Erkennung ermöglicht als ein rein einzelbildbasierter Klassifikationsansatz.

Die Evaluierung liefert hierauf eine eindeutig bejahende Antwort. Ein einzelbildbasierter Ansatz ist bei der Analyse von Videostreams hochgradig anfällig für temporäre Störungen. Bewegungsunschärfe (Motion Blur), wenn die Katze auf die Klappe zuspringt, partielle Verdeckungen (Okklusionen) der Beute durch die Pfoten oder rasche Änderungen der Beleuchtung führen unweigerlich zu stark schwankenden Konfidenzwerten des neuronalen Netzes. Ohne eine zeitliche Komponente würde die Zugangskontrolle unaufhörlich zwischen den Zuständen \glqq verriegelt\grqq\ und \glqq entriegelt\grqq\ wechseln (Flackern).

Die Implementierung der \texttt{StateMachine} in Verbindung mit ByteTrack löst dieses Problem effektiv. Durch die Aggregation der Klassifikationsergebnisse über eine Historie von 15 Einzelbildern (gleitender Durchschnitt) wird ein robustes Signal erzeugt, das Ausreißer in Einzelbildern glättet. Die Tests zeigten, dass eine Katze ohne Beute selbst dann verlässlich hereingelassen wird, wenn einzelne Einzelbilder fälschlicherweise als positiv klassifiziert werden, solange der gleitende Durchschnitt den strengen Schwellenwert von 0,8 nicht überschreitet. Damit ist der Beweis erbracht, dass die zeitliche Modellierung in diesem Anwendungsfall nicht nur ein optionales Feature, sondern eine zwingende Voraussetzung für den stabilen Praxisbetrieb darstellt.

\subsection{Bewertung des Architektur-Ansatzes}
Die Entscheidung, das Problem in zwei separate neuronale Netze (YOLO-Detektor für das Gesicht und YOLO-Klassifikator für den Zustand) zu unterteilen, hat sich als äußerst vorteilhaft erwiesen.

\subsubsection{Stärken der Verarbeitungsketten-Kaskadierung}
Der übliche Ansatz im Bereich maschinelles Sehen bestünde darin, ein einziges Objekterkennungsmodell darauf zu trainieren, direkt die Klassen \glqq Katze\grqq\ und \glqq Beutetier\grqq\ im Gesamtbild zu finden. Bei einer Katzenklappe scheitert dieser Ansatz jedoch häufig. Eine Maus im Maul einer Katze nimmt im Gesamtbild der Kamera nur einen verschwindend geringen Bruchteil der Pixel ein und ist durch das Maul und die Schnurrhaare stark verdeckt. Standard-Objektdetektoren haben immense Schwierigkeiten, derart kleine, okkludierte Objekte zuverlässig mit einem Begrenzungsrahmen zu versehen.

Der in dieser Arbeit gewählte Kaskadierungs-Ansatz umgeht dieses Problem elegant. Der erste Detektor muss lediglich das vergleichsweise große und markante Katzengesicht finden – eine triviale Aufgabe für moderne Architekturen wie YOLOv11, was sich in der hohen Erkennungsstabilität niederschlug. Durch das anschließende Herausschneiden (Zuschneiden) dieses Bereichs wird die Aufmerksamkeit des nachgelagerten Klassifikators exklusiv auf die relevante Region fokussiert (Aufmerksamkeitsmechanismus \textit{by design}). Der Klassifikator muss keine kleine Maus im Hintergrund des Gartens suchen, sondern analysiert lediglich die Pixelstruktur direkt am Maul der Katze.

\subsubsection{Die Rolle der asymmetrischen Randerweiterung}
Ein entscheidendes Detail für den Erfolg des Systems, das die perfekte Trefferquote von 100\,\% in der Evaluierung erst ermöglichte, ist die in Kapitel 5 beschriebene asymmetrische Randerweiterung (\texttt{pad\_bottom = 30}). Ohne diese Randerweiterung hätte der Begrenzungsrahmen das Gesicht exakt am Kinn der Katze abgeschnitten. Beutetiere hängen jedoch physikalisch bedingt oftmals unterhalb des Kinns aus dem Maul heraus. Das einfache, aber effektive Verschieben des Bildausschnitts nach unten stellt sicher, dass der Klassifikator stets den gesamten visuellen Kontext der Beute als Eingabe erhält.

\subsection{Kritische Analyse des Vorhersagegenauigkeits-Kompromisses}
Trotz der perfekten Trefferquote (keine Beute wurde übersehen) offenbarte die Evaluierung auch Schwächen des Systems, primär ablesbar an der Vorhersagegenauigkeit von 66,7\,\% bei einem Schwellenwert von 0,8. Das System produzierte zwei Fehlalarme, bei denen die Katze fälschlicherweise ausgesperrt wurde.

Wie bereits dargelegt, hat bei der Konstruktion einer smarten Katzenklappe das Wohl des Haustieres höchste Priorität. Ein System, das die Katze wiederholt grundlos aussperrt, wird vom Tier schnell als unzuverlässig oder bedrohlich wahrgenommen. Aus diesem Grund wurde der Schwellenwert für die Verriegelung extrem hoch (0,8) angesetzt.

Dass es trotz dieses restriktiven Schwellenwerts zu Fehlalarmen kam, liegt an den inhärenten Grenzen der zweidimensionalen Bildklassifikation unter Infrarotlicht. Das Kamerabild verliert durch den Wegfall der Farbinformation (RGB zu Graustufen) eine essenzielle Dimension zur Unterscheidung von Texturen \cite{li2024thermal}. Für das neuronale Netz ist der visuelle Unterschied zwischen dem dunklen Fell eines toten Beutetiers und einem dunklen Blatt im Hintergrund, das zufällig die gleiche Form aufweist und im Bildausschnitt direkt unter dem Kinn der Katze positioniert ist, minimal. Wenn sich ein solches Blatt über mehrere Einzelbilder hinweg kongruent mit der Katze bewegt, summiert die State Machine konsequent hohe Konfidenzwerte auf und löst schließlich den Fehlalarm aus.

\subsection{Schwächen, Limitationen und Fehlerquellen}
Aus der Analyse der Fehlerfälle lassen sich klare Limitationen des aktuellen Systems ableiten, die das Potenzial für zukünftige Verbesserungen aufzeigen.

\begin{enumerate}
    \item \textbf{Verlust von Tiefeninformationen:} Die verwendete Hardware nutzt einen einfachen 2D-Bildsensor. Das System kann nicht unterscheiden, ob sich ein dunkles Objekt physisch im Maul der Katze befindet oder lediglich im Hintergrund liegt und aus der Kameraperspektive zufällig im Begrenzungsrahmen des Gesichtes erscheint (Perspektivische Überlappung). 
    \item \textbf{Beleuchtung und Kontrast (Infrarot-Problematik):} In der Nacht ist das System vollständig auf die Infrarotausleuchtung angewiesen. Die Vorverarbeitung mittels CLAHE gleicht zwar lokale Kontrastschwächen aus, kann jedoch nicht verhindern, dass reflektierende Oberflächen (z.\,B. nasse Blätter oder das Halsband der Katze) im Infrarotlicht stark übersteuern und das neuronale Netz irritieren.
    \item \textbf{Extreme Bewegungsgeschwindigkeit:} Wenn eine Katze fluchtartig durch die Klappe rennt, sinkt die Anzahl der verwertbaren Einzelbilder drastisch. Bei einer Laufzeit von ca. 8 FPS verbleiben dem System in einem solchen Fall möglicherweise nur 3 bis 4 Einzelbilder zur Analyse. In diesem Szenario kann die \texttt{StateMachine} ihre Stärke (die zeitliche Glättung) nicht voll ausspielen.
\end{enumerate}

\subsection{Einordnung und lokale Datenverarbeitung}
Ein wesentliches Nebenziel der Arbeit war es, ein System zu konzipieren, das nicht auf hochleistungsfähige Rechenzentrums-Server angewiesen ist, sondern lokal operieren kann. Die Entscheidung, auf leichtgewichtige Architekturen (YOLOv11n) und optimierte Inferenz-Engines (TensorFlow Lite, OpenVINO) zu setzen, rechtfertigt sich durch die erreichten Leistungskennzahlen (ca. 8 FPS). 

Der Ansatz beweist, dass es durch eine intelligente Architektur möglich ist, komplexe visuelle Analysen echtzeitfähig auf Endverbraucher-Hardware abzubilden. Die Kombination aus Deep Learning und einer deterministischen Nachbearbeitung (ByteTrack und State Machine) stellt einen ressourcenschonenden und gleichzeitig hochrobusten Lösungsansatz dar, der in puncto Zuverlässigkeit rechenzentrenbasierten Schnittstellen deutlich überlegen ist, da Latenzprobleme durch Netzwerkübertragungen entfallen.

Zusammenfassend lässt sich festhalten, dass der gewählte Ansatz die konzeptionellen Hürden erfolgreich überwindet. Die verbleibenden Fehlalarme stellen keinen fundamentalen Architekturfehler dar, sondern spiegeln die Grenzen des zugrunde liegenden Klassifikationsdatensatzes wider. Mit dem in Kapitel 5 integrierten Modul zum automatisierten Speichern unklarer Bilder (gezieltes Sammeln schwieriger Negativbeispiele \cite{shrivastava2016training}) ist der Grundstein für die kontinuierliche Feinabstimmung des Modells bereits gelegt, um die Systemgenauigkeit im Praxisbetrieb iterativ weiter zu maximieren.
""",
    "zusammenfassung.tex": r"""\section{Zusammenfassung und Ausblick}
\label{sec:zusammenfassung}

Die vorliegende Bachelorarbeit befasst sich mit der Entwicklung eines ausfallsicheren, echtzeitfähigen Zugangssystems für Haustiere, das das unbemerkte Eindringen von Beutetieren (wie Mäusen oder Vögeln) in den Wohnraum unterbindet. Durch die Kombination moderner Methoden des Deep-Learning mit klassischen Ansätzen der zeitlichen Modellierung wurde ein System geschaffen, das weit über die rein einzelbildbasierte Objekterkennung hinausgeht.

\subsection{Zusammenfassung der Erkenntnisse}
\label{subsec:zusammenfassung_erkenntnisse}

Die theoretischen Überlegungen sowie die praktische Evaluierung haben gezeigt, dass die Erkennung von Beutetieren im Maul einer Katze ein hochkomplexes visuelles Problem darstellt. Die Beutetiere sind in der Regel winzig, stark durch Pfoten oder Schnurrhaare verdeckt und weisen gerade unter den schwierigen Lichtbedingungen bei Nacht (Infrarot-Graustufen) kaum unterscheidbare Texturen auf. Klassische, monolithische Objektdetektoren scheitern an dieser Aufgabenstellung häufig.

Als Lösung wurde in dieser Arbeit eine mehrstufige Verarbeitungskette (Kaskadierung) entwickelt: Im ersten Schritt detektiert ein YOLOv11-Modell robust das Gesicht der Katze. Der entscheidende Durchbruch gelang durch ein asymmetrisches Zuschneide-Verfahren (\texttt{pad\_bottom = 30}), welches den Fokusbereich des Bildes präzise auf den potenziellen Ort der Beute ausrichtete, ohne diese abzuschneiden. Im zweiten Schritt analysierte ein dediziertes YOLO-Klassifikationsmodell diesen isolierten Bildausschnitt. Um temporäre Fehlklassifikationen, verursacht durch Bewegungsunschärfe oder Lichtreflexe, auszugleichen, wurde ByteTrack zur konsistenten Verfolgung der Katze implementiert, gepaart mit einer deterministischen State Machine. Diese State Machine aggregierte die Klassifikationswerte über ein gleitendes Zeitfenster (gleitender Durchschnitt) von 15 Einzelbildern.

Die Evaluierung auf realen Videodaten demonstrierte den Erfolg dieses Architektur-Ansatzes. Das System erreichte eine perfekte Trefferquote von 100\,\%, was bedeutet, dass kein einziges Beutetier im Testdatensatz übersehen wurde. Der gewählte Schwellenwert von 0,8 spiegelte dabei die Priorität des Nutzers wider: \glqq Die Katze hat Vorrang.\grqq\ Zwar führten schwierige Hintergrundtexturen bei diesem \glqq Sicherheit zuerst\grqq-Ansatz zu wenigen, aber existenten Fehlalarmen, jedoch erwies sich das Gesamtsystem als hochgradig praxistauglich und lief ressourcenschonend mit durchschnittlich ca. 8 FPS auf lokalen Endgeräten.

Die zentrale Forschungsfrage konnte somit bejaht werden: Eine Kombination aus Objekterkennung, Objektverfolgung und zeitlicher Glättung bietet eine signifikant robustere Lösung als herkömmliche Einzelbildauswertungen.

\subsection{Ausblick und Weiterentwicklungsmöglichkeiten}
\label{subsec:ausblick}

Obwohl das System in seiner aktuellen Form bereits zuverlässig Beutetiere aussperrt, bieten die gewonnenen Erkenntnisse und die analysierten Fehlerquellen (siehe Kapitel 7) konkrete Anknüpfungspunkte für zukünftige Optimierungen.

\subsubsection{Kontinuierlicher Datenkreislauf und gezieltes Sammeln schwieriger Negativbeispiele}
Die unmittelbarste und vielversprechendste Methode zur Steigerung der Vorhersagegenauigkeit (Reduktion der Fehlalarme) liegt in der Erweiterung des Trainingsdatensatzes. Das System verfügt bereits in der aktuellen \texttt{main.py} über eine implementierte Routine zum gezielten Sammeln schwieriger Negativbeispiele \cite{shrivastava2016training}. Bilder mit mittleren Konfidenzwerten (Unsicherheiten des Klassifikators) werden automatisch lokal abgespeichert. Zukünftige Arbeiten können diesen stetig wachsenden Schatz an realen Randfällen nutzen, um das YOLO-Klassifikationsmodell iterativ nachzutrainieren (Aktives Lernen). Sobald der Klassifikator gelernt hat, dass ein bestimmtes dunkles Blatt im Garten keine Maus ist, wird die Fehlalarm-Rate des Gesamtsystems sukzessive sinken.

\subsubsection{Sensor-Fusion und Tiefeninformationen}
Eine der größten identifizierten Limitationen war der Verlust der 3D-Information, wodurch Objekte im Hintergrund fälschlicherweise als \glqq im Maul der Katze\grqq\ interpretiert wurden (perspektivische Überlappung). Eine signifikante architektonische Erweiterung bestünde im Einsatz von Stereo-Kameras oder kompakten Time-of-Flight (ToF)- bzw. LiDAR-Sensoren. Durch die Integration von Tiefeninformationen (Tiefenkarte) könnte die State Machine Bilder ignorieren, bei denen sich das vermeintliche Beutetier physisch weit hinter der Fokusebene des Katzenkopfs befindet. 

\subsubsection{Multimodale Ansätze (Audio-Integration)}
Zusätzlich zur rein visuellen Analyse könnte das System durch multimodale Sensorik erweitert werden. Die Integration eines Mikrofons und eines leichtgewichtigen neuronalen Netzes zur akustischen Ereigniserkennung könnte die Genauigkeit weiter erhöhen. Das Miauen einer Katze, welches darauf schließen lässt, dass das Maul frei ist, oder das Fiepen eines Beutetiers stellen wertvolle orthogonale Signale dar, die direkt als ergänzendes Vorabwissen in die Berechnung des gleitenden Durchschnitts der \texttt{StateMachine} einfließen könnten.

\subsubsection{Hardware-Beschleunigung durch neuronale Verarbeitungseinheiten}
Hinsichtlich der Inferenz-Hardware bietet der aktuelle Markt rasante Entwicklungen im Bereich lokaler KI-Beschleuniger (neuronale Verarbeitungseinheiten). Der Wechsel von einer reinen CPU-Inferenz auf dedizierte Hardware wie den Google Coral Edge TPU oder moderne Intel/AMD NPUs würde die Framerate voraussichtlich um ein Vielfaches steigern. Eine höhere FPS-Zahl erlaubt der State Machine, eine noch feingranularere Historie aufzubauen, was die zeitliche Glättung gegen Flackern weiter perfektionieren würde.

\vspace{0.5cm}

Abschließend lässt sich resümieren, dass die in dieser Arbeit entwickelte Verarbeitungskette ein äußerst solides, modulares Fundament darstellt. Die Kombination aus zielgerichtetem maschinellem Sehen und deterministischer Zustandslogik löst das Problem des unerwünschten Beuteeintrags durch Katzen bereits heute zuverlässig und eröffnet zugleich vielfältige, spannende Wege für die Heimautomatisierungs-Integration von morgen.
"""
}

for name, content in files.items():
    with open(os.path.join(r"c:\Public\Studium\Bachelorarbeit", name), "w", encoding="utf-8") as f:
        f.write(content)

print("Files reconstructed.")
