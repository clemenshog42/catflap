Hier ist das komplette Markdown-Dokument mit allen Fragen und den dazugehörigen Antworten aus deinen Unterlagen. Ich habe HTML-Tags (`<details>` und `<summary>`) verwendet, damit die Antworten in Markdown-Viewern einklappbar sind.

Du kannst den Codeblock einfach kopieren und als `.md`-Datei (z. B. `Klausurfragen_CGB.md`) speichern.

```markdown
# Klausurvorbereitung: Computergrafik & Bildverarbeitung

Dieses Dokument enthält die gesammelten Fragen und Antworten aus den Altklausuren (inklusive Sommer 2019 und Sommer 2024). Klicke auf die jeweilige Frage, um die Antwort auszuklappen.

---

## Teil 1: Computergrafik (CG)

<details>
<summary><b>Frage:</b> Nenne die 3 Hauptschritte der Grafikpipeline in der korrekten Reihenfolge und beschreibe sie kurz.</summary>
<br>
<b>Antwort:</b> 
1. <b>Application:</b> Interaktion, Animation und das Datenmodell werden verarbeitet.
2. <b>Geometry:</b> Transformationen (3D-Daten) und Projektionen werden berechnet.
3. <b>Rasterization:</b> Die Projektion der Daten auf die 2D-Ebene (Pixel) findet statt.
</details>

<details>
<summary><b>Frage:</b> Wofür können Normalvektoren (Surface Normals) verwendet werden? Nenne zwei Beispiele.</summary>
<br>
<b>Antwort:</b> 
1. <b>Erkennen der Außenseite:</b> Wichtig für das Backface Culling, um zu bestimmen, welche Seite eines Polygons der Kamera zugewandt ist.
2. <b>Beleuchtung:</b> Wird im Beleuchtungsmodell benötigt, um zu berechnen, wie das Licht an der Oberfläche reflektiert wird (z. B. im Phong-Modell).
</details>

<details>
<summary><b>Frage:</b> Nenne drei verschiedene Entfernungsverfahren (Culling) und beschreibe sie.</summary>
<br>
<b>Antwort:</b> 
1. <b>Backface Culling:</b> Rückseiten von Objekten (die von der Kamera abgewandt sind) werden nicht gerendert.
2. <b>Occlusion Culling:</b> Objekte, die komplett von anderen Objekten verdeckt werden, werden nicht gerendert.
3. <b>Frustum Culling:</b> Objekte, die sich nicht im sichtbaren Kamerabereich (Viewing Frustum) befinden, werden nicht gerendert.
</details>

<details>
<summary><b>Frage:</b> Wofür stehen die Vektoren $L$ und $N$ im Phong-Beleuchtungsmodell?</summary>
<br>
<b>Antwort:</b> 
* $L$ zeigt in Richtung der Lichtquelle (Light).
* $N$ ist die Oberflächennormale (Surface Normal).
</details>

<details>
<summary><b>Frage:</b> Welche drei Licht- bzw. Reflexionsarten werden im Phong-Modell addiert?</summary>
<br>
<b>Antwort:</b> Ambient (Umgebungslicht), Diffuse (matte Streuung) und Specular (Glanzpunkte).
</details>

<details>
<summary><b>Frage:</b> Welcher Texturtyp eignet sich für Displacement Mapping bzw. Bump Mapping und welcher für Normal Mapping?</summary>
<br>
<b>Antwort:</b> 
* <b>Displacement / Bump Mapping:</b> Graustufen-Textur (enthält nur Tiefen-/Höheninformationen).
* <b>Normal Mapping:</b> RGB-Textur (Die drei Farbkanäle repräsentieren die x-, y- und z-Koordinaten der Normalen).
</details>

<details>
<summary><b>Frage:</b> Wie stellt man im Raytracing-Algorithmus fest, ob ein Punkt im Schatten liegt?</summary>
<br>
<b>Antwort:</b> Man sendet einen Strahl (Shadow Ray) von dem Punkt zur Lichtquelle. Besteht keine direkte "Line of Sight" (weil der Strahl ein anderes Objekt trifft), liegt der Punkt im Schatten.
</details>

<details>
<summary><b>Frage:</b> Worin unterscheiden sich Partikelsysteme (Particle Systems) von Schwarmalgorithmen (Flocking) grundlegend?</summary>
<br>
<b>Antwort:</b> 
* <b>Partikelsysteme:</b> Verhalten sich unabhängig von ihren Nachbarn und haben meist eine definierte Lebensdauer (Life span).
* <b>Schwarmalgorithmen:</b> Kennen Position und Richtung der Nachbarn (interagieren miteinander) und haben keine klassische Lebensdauer (zeitunabhängig).
</details>

<details>
<summary><b>Frage:</b> Was ist der Z-Buffer (Depth Buffer)?</summary>
<br>
<b>Antwort:</b> Er speichert die Tiefeninformation (Z-Koordinate) jedes Pixels, um beim Rendern festzustellen, ob ein Polygon vor oder hinter einem bereits gezeichneten Polygon liegt.
</details>

---

## Teil 2: Bildverarbeitung (BV)

<details>
<summary><b>Frage:</b> Erläutere die beiden Konvertierungsstrategien "Lightness" und "Luminosity" von RGB nach Grauwert.</summary>
<br>
<b>Antwort:</b> 
1. <b>Lightness:</b> Berechnet sich durch $g = (\max(R,G,B) + \min(R,G,B)) / 2$. Dies gleicht lediglich die statistische Mittelwertverschiebung aus.
2. <b>Luminosity:</b> Berechnet sich durch $g = 0.21R + 0.72G + 0.07B$. Dieses Verfahren ist an die menschliche Wahrnehmung angelehnt, da das menschliche Auge für grünes Licht am empfindlichsten ist.
</details>

<details>
<summary><b>Frage:</b> Was besagt das Nyquist'sche Abtasttheorem?</summary>
<br>
<b>Antwort:</b> Um ein periodisches Signal messen und fehlerfrei (ohne Aliasing) rekonstruieren zu können, muss es mit mindestens der doppelten maximalen Frequenz ($f_{max}$) abgetastet werden.
</details>

<details>
<summary><b>Frage:</b> Nenne die beiden zentralen Operatoren der mathematischen Morphologie.</summary>
<br>
<b>Antwort:</b> Erosion (Abtragung) und Dilatation (Erweiterung).
</details>

<details>
<summary><b>Frage:</b> Sortiere die Distanzmetriken Euklid, Manhattan und Schachbrett nach der Länge ihrer Pfade (absteigend).</summary>
<br>
<b>Antwort:</b> Manhattan (Längster Pfad) > Euklid > Schachbrett (Kürzester Pfad).
</details>

<details>
<summary><b>Frage:</b> Nenne Möglichkeiten zur Ausnahmebehandlung der Randbereiche bei der Filterung im Ortsraum.</summary>
<br>
<b>Antwort:</b> 
1. Bounds überprüfen (Rand ignorieren).
2. Rahmen hinzufügen (Padding).
3. Outputbild verkleinern.
</details>

<details>
<summary><b>Frage:</b> Erläutere den Begriff "Convolution Theorem" (Faltungssatz).</summary>
<br>
<b>Antwort:</b> Eine Faltung (Convolution) im Ortsraum entspricht mathematisch exakt einer Multiplikation im Frequenzraum (Fourierraum).
</details>

<details>
<summary><b>Frage:</b> Was ist der Unterschied zwischen Lokalisierung, Klassifikation und Segmentierung?</summary>
<br>
<b>Antwort:</b> 
* <b>Lokalisierung:</b> Finden eines Objektes im Bild (z.B. per Bounding Box).
* <b>Klassifikation:</b> Ein ganzes Bild durch ein Schlagwort beschreiben.
* <b>Segmentierung:</b> Erkennen von zusammengehörigen Objekten im Bild durch die exakte Zuweisung von Pixeln zu einem Objekt.
</details>

<details>
<summary><b>Frage:</b> Definiere Entropie, Homogenität und psychovisuelle Redundanz im Kontext von Bilddaten.</summary>
<br>
<b>Antwort:</b> 
* <b>Entropie:</b> Maß für die Unordnung oder das Fehlen von Vorhersagbarkeit in den Daten.
* <b>Homogenität:</b> Gleichartigkeit von Daten (z. B. große Flächen mit gleicher Farbe).
* <b>Psychovisuelle Redundanz:</b> Informationen im Bild, die für den Menschen visuell ohnehin nicht wahrnehmbar sind (z.B. feinste Farbnuancen) und bei der Kompression weggelassen werden können.
</details>

<details>
<summary><b>Frage:</b> Welches Kompressionsverfahren eignet sich gut für hohe Homogenität und welches für hohe Entropie?</summary>
<br>
<b>Antwort:</b> 
* Für <b>Homogenität</b>: Runlength Coding (Lauflängenkodierung).
* Für <b>Entropie</b>: Huffman Coding.
* (LZW-Kompression ist ein Hybrid und nutzt beides.)
</details>

<details>
<summary><b>Frage:</b> Warum beschleunigt eine Distance Map (DM) den ICP-Registrierungsalgorithmus?</summary>
<br>
<b>Antwort:</b> ICP arbeitet iterativ. Durch eine vorausberechnete Distance Map muss der Abstand zu den Punkten nicht in jeder einzelnen Iteration für alle Kombinationen neu berechnet werden. Der Zugriff auf die DM erfolgt in $O(n)$ statt $O(n \cdot m)$.
</details>

```