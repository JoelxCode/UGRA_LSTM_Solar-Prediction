import nbformat
from nbformat.v4 import new_code_cell
from nbclient import NotebookClient

nb_path = 'prediction (newest).ipynb'
nb = nbformat.read(nb_path, as_version=4)

code = r"""
# This cell prints the contents of 'Skip to content document'
import os

fname_base = 'Skip to content document'
candidates = [fname_base, fname_base + '.txt', fname_base + '.md', fname_base + '.html', fname_base + '.ipynb']

found = None
for f in candidates:
    if os.path.exists(f):
        found = f
        break

if not found:
    raise FileNotFoundError("Could not find any of: " + ", ".join(candidates))

if found.endswith('.ipynb'):
    import nbformat
    nb2 = nbformat.read(found, as_version=4)
    for i, cell in enumerate(nb2.cells):
        print(f"--- cell {i} ({cell.cell_type}) ---")
        print(cell.source)
        print()
else:
    with open(found, 'r', encoding='utf-8') as fh:
        print(fh.read())
"""

cell = new_code_cell(code)
nb.cells.insert(0, cell)

# execute the notebook so the output gets captured into the notebook file
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
nbformat.write(nb, nb_path)
print(f"Inserted and executed printing cell; outputs saved in {nb_path}.")