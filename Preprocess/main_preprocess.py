"""
Preprocessing of a Nastran input file (*.bdf / *.dat / *.nas): reads the model, analyses it and shows it in 3D.

    python Preprocess/main_preprocess.py                    -> dialog to choose the BDF file
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

from Preprocess.bdf_model import BdfModel  # noqa: E402
from Preprocess.bdf_viewer import BdfViewer  # noqa: E402
from visualization.h5_viewer import qt_app  # noqa: E402
from qtpy import QtWidgets  # noqa: E402

# fixed input file (None = file dialog / command line argument), e.g. ROOT / "edit.bdf"
BDF_FILE = None


def choose_file():
    qt_app()                                                    # Qt application must exist before the dialog
    path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Nastran input file", str(ROOT),
                                                    "Nastran input (*.bdf *.dat *.nas *.blk);;All files (*)")
    return path or None


if __name__ == "__main__":
    bdf_file = Path(sys.argv[1]) if len(sys.argv) > 1 else BDF_FILE or choose_file()
    if bdf_file is None:
        sys.exit("No file selected.")

    model = BdfModel(bdf_file)
    print(model.report)
    BdfViewer(model).show()
