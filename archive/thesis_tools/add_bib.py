import io

bib_entry = """
@misc{surepetcare_microchip,
  author = {{Sure Petcare}},
  title = {Microchip Cat Flap},
  year = {2026},
  note = {Verf{\\"u}gbar unter: \\url{https://www.surepetcare.com/en-gb/pet-doors/microchip-cat-flap}}
}
"""

with io.open('C:/Public/Studium/Bachelorarbeit/thesis/literatur.bib', 'a', encoding='utf-8') as f:
    f.write(bib_entry)
print('BibTeX entry added successfully.')
