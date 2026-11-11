"""
3D visualization of MSC Nastran HDF5 results (*.h5).

Usage:
    python main_h5.py                       -> window with Open / recent files (or H5_FILE below if it is set)
    python main_h5.py path/to/results.h5    -> opens the given file

Static results, load / time steps, frequency response (complex results) and modes are supported. Select a node
by a double-click in 3D (or its ID) to see its values; for more steps a graph over time / frequency is shown.
Frequency response: tab "Random" = random vibration (response PSD, RMS / 3 sigma maps), tab "Modes" = natural
frequencies; Animate = animation of the deformed shape.

The window has the menu File (Open, Open recent, Reload, Save camera view), files can be dragged & dropped into it
and they are loaded in the background. Also input files (*.bdf, *.dat, *.nas) are opened (Preprocess viewer).

The viewer is a Qt window, it needs (once):  pip install pyvistaqt PySide6   (+ matplotlib for the graph)
"""
import sys
from pathlib import Path

from file_processing.read_Nastran_files import ParseH5File
from visualization.viewer_window import open_viewer

ROOT = Path(__file__).resolve().parent

# file opened at start, None = empty window (File > Open, drag & drop, recent files) / command line argument
# e.g. ROOT / "file_processing" / "box2_weight1g_z_ch.h5" or ROOT / "random_ram_edit.h5"
H5_FILE = None

# optionally save mesh + results of the first step to *.vtu (can be opened in ParaView), None = do not save
VTU_FILE = None   # e.g. ROOT / "results.vtu"


if __name__ == "__main__":
    h5_file = Path(sys.argv[1]) if len(sys.argv) > 1 else H5_FILE

    if VTU_FILE is not None and h5_file is not None:
        h5 = ParseH5File(h5_file)
        h5.print_summary()
        h5.save_vtu(VTU_FILE)
        open_viewer(model=h5)                                   # already loaded
    else:
        open_viewer(h5_file)                                    # loaded in the background (progress in status bar)
