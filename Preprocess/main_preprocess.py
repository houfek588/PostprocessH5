"""
Preprocessing of a Nastran input file (*.bdf / *.dat / *.nas): reads the model, analyses it and shows it in 3D.

    python Preprocess/main_preprocess.py                    -> empty window (File > Open, drag & drop, recent)
    python Preprocess/main_preprocess.py path/to/model.bdf  -> given file
    (or set BDF_FILE below)

Window: tab "Model" = table of model fields (Property ID, Material ID, element type, connected bodies, contact
bodies, volume, mass, element quality), model entities (SPC, SPCD, RBE, CONM2, GRAV, FORCE), sections, probe;
tab "Report" = text report (summary, materials, properties, mass + CG, case control with reference check,
constraints, loads, dynamic cards, rigid elements, contact, mesh checks). The report is printed to the console too.

Requires: pyNastran, scipy, pyvista, pyvistaqt + PySide6 (or PyQt5)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.viewer_window import open_viewer  # noqa: E402

# fixed input file (None = empty window: File > Open, drag & drop, recent files / command line argument),
# e.g. ROOT / "edit.bdf"
BDF_FILE = None


if __name__ == "__main__":
    bdf_file = Path(sys.argv[1]) if len(sys.argv) > 1 else BDF_FILE
    # the model is read and analysed in the background (progress in the status bar), the report is printed to the
    # console and shown in the tab "Report"; other files: File > Open or drag & drop (also H5 results)
    open_viewer(bdf_file)
