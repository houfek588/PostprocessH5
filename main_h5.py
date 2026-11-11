"""
3D visualization of MSC Nastran HDF5 results (*.h5).

Usage:
    python main_h5.py                       -> dialog to choose the H5 file (or H5_FILE below if it is set)
    python main_h5.py path/to/results.h5    -> opens the given file

Static results, load / time steps, frequency response (complex results) and modes are supported. Select a node
by a double-click in 3D (or its ID) to see its values; for more steps a graph over time / frequency is shown.

The viewer is a Qt window, it needs (once):  pip install pyvistaqt PySide6   (+ matplotlib for the graph)
"""
import sys
from pathlib import Path

from file_processing.read_Nastran_files import ParseH5File
from visualization.h5_viewer import H5ResultViewer, qt_app
from qtpy import QtWidgets

ROOT = Path(__file__).resolve().parent

# fixed input file, None = dialog / command line argument
# e.g. ROOT / "file_processing" / "box2_weight1g_z_ch.h5" or ROOT / "random_ram_edit.h5"
H5_FILE = None

# optionally save mesh + results of the first step to *.vtu (can be opened in ParaView), None = do not save
VTU_FILE = None   # e.g. ROOT / "results.vtu"


def choose_file():
    qt_app()                                                    # Qt application must exist before the dialog
    path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Nastran H5 results", str(ROOT),
                                                    "Nastran HDF5 results (*.h5);;All files (*)")
    return path or None


if __name__ == "__main__":
    h5_file = Path(sys.argv[1]) if len(sys.argv) > 1 else H5_FILE or choose_file()
    if h5_file is None:
        sys.exit("No file selected.")

    h5 = ParseH5File(h5_file)
    h5.print_summary()

    if VTU_FILE is not None:
        h5.save_vtu(VTU_FILE)

    H5ResultViewer(h5).show()
