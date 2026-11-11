"""
Interactive 3D viewer of MSC Nastran H5 results (read by file_processing.read_Nastran_files.ParseH5File).

Qt window:
    left   - result step (time / frequency / mode) with amplitude / phase / real / imaginary part of complex
             results, table of all results (click / arrow keys = show the result), display and section settings,
             selected node (double-click in 3D or node ID) with the values of the current result group
    right  - 3D view (pyvista) and, if the results have more steps, a graph of the current result in the selected
             node over time / frequency (click into the graph = show that step)

Requires Qt for Python (once):   pip install pyvistaqt PySide6        (PyQt5 / PyQt6 work as well)
If more Qt bindings are installed, PySide6 is used (can be changed by the environment variable QT_API).
The graph needs matplotlib.

3D view - mouse:  left = rotate, middle or Shift + left = pan, wheel or right = zoom, double-click = select node
3D view - keys:   I                  select the node under the mouse cursor
                  Left / Right       previous / next result
                  PageUp / PageDown  previous / next result step (time, frequency, mode)
                  X / Y / Z          section plane perpendicular to the X / Y / Z axis (switches the section on)
                  N                  flip the section (show the other part of the model)
                  R / V              reset camera / isometric view

Sections are made on a linear copy of the mesh (corner nodes of quadratic elements), they are cut
in the undeformed shape and deformed afterwards.
"""
import csv
import importlib.util
import os
import sys
import traceback
from collections import Counter

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter, vtkFeatureEdges
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkRenderingCore import vtkCellPicker

# PySide6 is preferred (tested; PyQt aborts the whole application on a Python error in a Qt callback)
if not os.environ.get("QT_API") and importlib.util.find_spec("PySide6") is not None:
    os.environ["QT_API"] = "pyside6"

try:
    from pyvistaqt import MainWindow, QtInteractor
    from qtpy import QtCore, QtGui, QtWidgets
except Exception as error:  # ImportError or "no Qt bindings found"
    raise ImportError("The H5 viewer needs Qt for Python, install it by:  pip install pyvistaqt PySide6") from error


def _qt_enum(scope, name):
    """Qt enum value working with PySide6 / PyQt5 / PyQt6 (unscoped or scoped access)."""
    value = getattr(QtCore.Qt, name, None)
    return value if value is not None else getattr(getattr(QtCore.Qt, scope), name)


_APP = None


def _print_qt_message(mode, context, message):
    # on Windows Qt writes its messages (also the fatal ones) to the debugger, not to the console -> print them
    print(f"Qt: {message}", file=sys.stderr)


def _print_python_error(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def qt_app():
    """
    Returns the QApplication - created once and kept alive (a QApplication without a Python reference is deleted
    immediately by PyQt and the next window crashes the program). Qt messages and Python errors in Qt callbacks
    are printed to the console.
    """
    global _APP
    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        QtCore.qInstallMessageHandler(_print_qt_message)
        if sys.excepthook is sys.__excepthook__:
            sys.excepthook = _print_python_error        # PyQt: print the error instead of aborting the program
    return _APP


USER_ROLE = _qt_enum("ItemDataRole", "UserRole")
HORIZONTAL = _qt_enum("Orientation", "Horizontal")
VERTICAL = _qt_enum("Orientation", "Vertical")
ITEM_ENABLED = _qt_enum("ItemFlag", "ItemIsEnabled")
TEXT_SELECTABLE = _qt_enum("TextInteractionFlag", "TextSelectableByMouse")

COMPLEX_MODES = (("amplitude", "Amplitude"), ("phase", "Phase [deg]"), ("real", "Real part"),
                 ("imag", "Imaginary part"))


def _fmt(value):
    return "-" if value is None or not np.isfinite(value) else f"{value:.4e}"


class H5ResultViewer:
    def __init__(self, h5, cmap="jet", show_edges=True, window_size=(1650, 980)):
        """
        Args:
            h5 (ParseH5File / NastranMesh): loaded Nastran results (or model)
            cmap (str): matplotlib colormap name
            show_edges (bool): show element edges at start
            window_size (tuple): window size in pixels
        """
        self.app = qt_app()

        self.h5 = h5
        h5.to_pyvista()                                 # fills h5.results
        self.results = h5.results                       # [(name, "point" / "cell")]
        self.steps = list(getattr(h5, "steps", []))     # result steps (time, frequency, modes)
        self.has_complex = any(h5.is_complex(name) for name, _ in self.results)
        per_subcase = Counter((s["subcase"], s["kind"]) for s in self.steps)
        self.has_history = any(n > 1 for n in per_subcase.values())

        # linear copy of the mesh (same points, cells and data arrays) with own data attributes
        self.grid = pv.UnstructuredGrid()
        self.grid.ShallowCopy(h5.to_pyvista(linear=True))
        self.grid.cell_data["_cell_index"] = np.arange(self.grid.n_cells)

        self.cmap = cmap
        self.show_edges = show_edges
        self.local_range = False                        # color range from the shown part / the whole model
        self.idx = 0                                    # index of the displayed result
        self.step = 0                                   # index of the displayed result step
        self.cmode = "amplitude"                        # shown part of complex results
        self.phase = 0.0                                # phase angle of the deformed shape of complex results
        self.deform_pct = 0.0                           # max deformation in % of the model size (at slider change)
        self.scale = 0.0                                # deformation scale factor (kept when the step changes)
        self.selected = None                            # (node index, cell index or None)
        self.curves = []                                # curves in the graph
        self._history_cache = {}
        self.model_size = self.grid.length              # diagonal of the bounding box

        self.section = False                            # section on / off
        self.slice_only = False                         # True = only the section plane, False = cut model
        self.plane_widget_on = True                     # interactive plane widget in the 3D view
        self.normal = np.array([0.0, 0.0, 1.0])         # section plane, normal points to the removed part
        self.origin = np.array(self.grid.center, dtype=float)

        # window: panel | (3D view / graph)
        self.window = MainWindow()
        self.window.setWindowTitle(f"Nastran results - {h5.file_path.name}")
        self.window.resize(*window_size)
        self.pl = QtInteractor()
        self.window.signal_close.connect(self.pl.close)
        if hasattr(h5, "close"):
            self.window.signal_close.connect(h5.close)
        self.pl.set_background("white")
        splitter = QtWidgets.QSplitter(HORIZONTAL)
        splitter.addWidget(self._build_panel())
        right = QtWidgets.QSplitter(VERTICAL)
        right.addWidget(self.pl.interactor)
        self.graph_widget = self._build_graph()
        if self.graph_widget is not None:
            right.addWidget(self.graph_widget)
            right.setSizes([window_size[1] - 300, 300])
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, window_size[0] - 440])
        self.window.setCentralWidget(splitter)

        self.actor = None
        self._bar_title = None
        self._build_surface()
        self._load_step()
        self._build_scene()
        self._update_tree_ranges()
        self._sync_controls()

    # ================================================================================================= Qt panel
    def _build_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        title = getattr(self.h5, "results_title", "results")
        layout.addWidget(QtWidgets.QLabel(f"<b>{self.h5.file_path.name}</b> - {title} "
                                          f"(click or arrow keys to show)"))

        # ---- result step (time / frequency / mode) + complex part
        self.step_group = QtWidgets.QGroupBox("Result step")
        grid = QtWidgets.QGridLayout(self.step_group)
        self.step_combo = QtWidgets.QComboBox()
        for s in self.steps:
            self.step_combo.addItem(s["label"])
        self.step_combo.currentIndexChanged.connect(lambda i: self.set_step(i))
        previous = QtWidgets.QPushButton("<")
        previous.setToolTip("Previous step (PageUp in 3D)")
        previous.setMaximumWidth(30)
        previous.clicked.connect(lambda checked=False: self.set_step(self.step - 1))
        following = QtWidgets.QPushButton(">")
        following.setToolTip("Next step (PageDown in 3D)")
        following.setMaximumWidth(30)
        following.clicked.connect(lambda checked=False: self.set_step(self.step + 1))
        self.step_slider = QtWidgets.QSlider(HORIZONTAL)
        self.step_slider.setRange(0, max(0, len(self.steps) - 1))
        self.step_slider.setTracking(False)                     # step is loaded after releasing the slider
        self.step_slider.valueChanged.connect(lambda v: self.set_step(v))
        self.step_slider.sliderMoved.connect(self._preview_step)
        grid.addWidget(previous, 0, 0)
        grid.addWidget(self.step_combo, 0, 1)
        grid.addWidget(following, 0, 2)
        grid.addWidget(self.step_slider, 1, 0, 1, 3)
        self.cmode_combo = QtWidgets.QComboBox()
        for _, label in COMPLEX_MODES:
            self.cmode_combo.addItem(label)
        self.cmode_combo.currentIndexChanged.connect(lambda i: self.set_complex_mode(COMPLEX_MODES[i][0]))
        self.phase_spin = QtWidgets.QSpinBox()
        self.phase_spin.setRange(0, 359)
        self.phase_spin.setSingleStep(15)
        self.phase_spin.setWrapping(True)
        self.phase_spin.setSuffix(" deg")
        self.phase_spin.setToolTip("Phase angle of the deformed shape (real part of U * exp(i * phase))")
        self.phase_spin.valueChanged.connect(lambda v: self.set_phase(v))
        complex_row = QtWidgets.QHBoxLayout()
        complex_row.addWidget(QtWidgets.QLabel("Complex:"))
        complex_row.addWidget(self.cmode_combo)
        complex_row.addWidget(QtWidgets.QLabel("deformation phase"))
        complex_row.addWidget(self.phase_spin)
        grid.addLayout(complex_row, 2, 0, 1, 3)
        for widget in [self.cmode_combo, self.phase_spin] + [complex_row.itemAt(i).widget() for i in (0, 2)]:
            widget.setVisible(self.has_complex)
        self.step_group.setVisible(len(self.steps) > 1 or self.has_complex)
        layout.addWidget(self.step_group)

        # ---- table of results
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Result", "Min", "Max"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(140)
        self._tree_items = {}
        parents = {}
        bold = QtGui.QFont()
        bold.setBold(True)
        for i, (name, location) in enumerate(self.results):
            group, item_label = self.h5.result_meta.get(name, ("Other", name))
            if group not in parents:
                parents[group] = self._group_item(self.tree, group, bold)
            item = QtWidgets.QTreeWidgetItem(parents[group], [item_label, "", ""])
            item.setData(0, USER_ROLE, i)
            kind = "nodal" if location == "point" else "element"
            item.setToolTip(0, f"{name}  ({kind} values" + (", complex)" if self.h5.is_complex(name) else ")"))
            self._tree_items[i] = item
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._on_tree_item)
        layout.addWidget(self.tree, stretch=1)

        # ---- display
        display = QtWidgets.QGroupBox("Display")
        grid = QtWidgets.QGridLayout(display)
        self.edges_check = QtWidgets.QCheckBox("Element edges")
        self.edges_check.setChecked(self.show_edges)
        self.edges_check.toggled.connect(self._on_edges)
        self.local_check = QtWidgets.QCheckBox("Color range of the shown part only")
        self.local_check.toggled.connect(self._on_local_range)
        self.deform_slider = QtWidgets.QSlider(HORIZONTAL)
        self.deform_slider.setRange(0, 100)                     # 0.1 % steps, 0 - 10 % of the model size
        self.deform_slider.valueChanged.connect(lambda v: self._on_deform_slider(v / 10.0))
        self.deform_label = QtWidgets.QLabel()
        deform_caption = QtWidgets.QLabel("Deformation:")
        grid.addWidget(self.edges_check, 0, 0)
        grid.addWidget(self.local_check, 0, 1)
        grid.addWidget(deform_caption, 1, 0)
        grid.addWidget(self.deform_slider, 1, 1)
        grid.addWidget(self.deform_label, 2, 0, 1, 2)
        if self.h5.get_deformation(0) is None:
            for widget in (deform_caption, self.deform_slider, self.deform_label):
                widget.hide()                                   # e.g. input model (BDF) without results
        layout.addWidget(display)

        # ---- section
        self.section_group = QtWidgets.QGroupBox("Section")
        self.section_group.setCheckable(True)
        self.section_group.setChecked(self.section)
        self.section_group.toggled.connect(lambda on: self.set_section(on))
        grid = QtWidgets.QGridLayout(self.section_group)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QLabel("Normal:"))
        for axis, normal in (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))):
            button = QtWidgets.QPushButton(axis)
            button.setToolTip(f"Section plane perpendicular to the {axis} axis (key {axis})")
            button.clicked.connect(lambda checked=False, n=normal: self.set_section_normal(n))
            buttons.addWidget(button)
        flip = QtWidgets.QPushButton("Flip")
        flip.setToolTip("Show the other part of the model (key N)")
        flip.clicked.connect(lambda checked=False: self.flip_section())
        buttons.addWidget(flip)
        grid.addLayout(buttons, 0, 0, 1, 2)
        self.section_slider = QtWidgets.QSlider(HORIZONTAL)
        self.section_slider.setRange(0, 1000)                   # 0.1 % steps
        self.section_slider.setTracking(False)                  # section is rebuilt after releasing the slider
        self.section_slider.valueChanged.connect(lambda v: self.set_section_position(v / 10.0))
        self.section_slider.sliderMoved.connect(
            lambda v: self.section_label.setText(f"Position: {v / 10.0:.1f} % (release to update)"))
        self.section_label = QtWidgets.QLabel()
        self.section_label.setWordWrap(True)
        grid.addWidget(QtWidgets.QLabel("Position:"), 1, 0)
        grid.addWidget(self.section_slider, 1, 1)
        grid.addWidget(self.section_label, 2, 0, 1, 2)
        self.slice_check = QtWidgets.QCheckBox("Slice only (show only the section plane)")
        self.slice_check.toggled.connect(lambda on: self.set_section(True, slice_only=on))
        self.plane_check = QtWidgets.QCheckBox("Plane widget in 3D (drag the arrow = rotate, sphere = move)")
        self.plane_check.setChecked(self.plane_widget_on)
        self.plane_check.toggled.connect(self._on_plane_check)
        grid.addWidget(self.slice_check, 3, 0, 1, 2)
        grid.addWidget(self.plane_check, 4, 0, 1, 2)
        layout.addWidget(self.section_group)

        # ---- selected node
        self.selection_group = QtWidgets.QGroupBox("Selected node (double-click in 3D, key I or node ID)")
        box = QtWidgets.QVBoxLayout(self.selection_group)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Node ID:"))
        self.node_edit = QtWidgets.QLineEdit()
        self.node_edit.setPlaceholderText("e.g. 1234")
        self.node_edit.returnPressed.connect(self._on_node_edit)
        row.addWidget(self.node_edit)
        show = QtWidgets.QPushButton("Show")
        show.clicked.connect(lambda checked=False: self._on_node_edit())
        row.addWidget(show)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda checked=False: self.select(None))
        row.addWidget(clear)
        box.addLayout(row)
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setWordWrap(True)
        self.selection_label.setTextInteractionFlags(TEXT_SELECTABLE)
        box.addWidget(self.selection_label)
        self.selection_table = QtWidgets.QTreeWidget()
        self.selection_table.setHeaderLabels(["Quantity", "Value"])
        self.selection_table.setRootIsDecorated(False)
        self.selection_table.setMaximumHeight(130)
        box.addWidget(self.selection_table)
        layout.addWidget(self.selection_group)

        # ---- view buttons
        buttons = QtWidgets.QHBoxLayout()
        for text, callback in (("Reset view", lambda: self.pl.reset_camera()),
                               ("Isometric", lambda: self.pl.view_isometric()),
                               ("Screenshot...", self._on_screenshot)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, c=callback: c())
            buttons.addWidget(button)
        layout.addLayout(buttons)

        help_label = QtWidgets.QLabel("3D view: left mouse = rotate, middle / Shift+left = pan, wheel = zoom, "
                                      "double-click = select node; keys I = select node, Left/Right = result, "
                                      "PageUp/PageDown = step, X/Y/Z = section, N = flip, R = reset view, "
                                      "V = isometric")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray")
        layout.addWidget(help_label)
        return panel

    def _preview_step(self, step):
        """Shows the label of the step while the slider is dragged (the step is loaded after releasing it)."""
        self.step_combo.blockSignals(True)
        self.step_combo.setCurrentIndex(step)
        self.step_combo.blockSignals(False)

    @staticmethod
    def _group_item(parent, text, font):
        item = QtWidgets.QTreeWidgetItem(parent, [text])
        item.setFlags(ITEM_ENABLED)                             # group header - not selectable
        item.setFont(0, font)
        return item

    def _update_tree_ranges(self):
        """Min / max of all results in the current step (and complex part)."""
        for i, (name, _) in enumerate(self.results):
            values = self._values(i)
            finite = values[np.isfinite(values)]
            low, high = (f"{finite.min():.4g}", f"{finite.max():.4g}") if finite.size else ("-", "-")
            self._tree_items[i].setText(1, low)
            self._tree_items[i].setText(2, high)
        for column in range(3):
            self.tree.resizeColumnToContents(column)

    def _sync_controls(self):
        """Updates the Qt controls according to the current state (without triggering their callbacks)."""
        widgets = (self.section_group, self.slice_check, self.section_slider, self.tree, self.step_combo,
                   self.step_slider, self.cmode_combo, self.phase_spin)
        for w in widgets:
            w.blockSignals(True)
        self.section_group.setChecked(self.section)
        self.slice_check.setChecked(self.slice_only)
        self.section_slider.setValue(int(round(self._section_fraction() * 10)))
        self.tree.setCurrentItem(self._tree_items[self.idx])
        if self.steps:
            self.step_combo.setCurrentIndex(self.step)
            self.step_slider.setValue(self.step)
        self.cmode_combo.setCurrentIndex([m for m, _ in COMPLEX_MODES].index(self.cmode))
        self.phase_spin.setValue(int(round(self.phase)) % 360)
        for w in widgets:
            w.blockSignals(False)
        self.section_label.setText(self._section_text())
        text = f"{self.deform_pct:.1f} % of model size"
        if self.scale > 0:
            text += f"  (scale {self.scale:.4g}x, kept for other steps)" if len(self.steps) > 1 else \
                f"  (scale {self.scale:.4g}x)"
        self.deform_label.setText(f"Deformation: {text}")

    def _on_tree_item(self, current, previous=None):
        idx = current.data(0, USER_ROLE) if current is not None else None
        if idx is not None:
            self.set_result(int(idx))

    def _on_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Save screenshot",
                                                        str(self.h5.file_path.with_suffix(".png")),
                                                        "PNG image (*.png)")
        if path:
            self.screenshot(path)

    # ================================================================================================= graph
    def _build_graph(self):
        """Graph of the current result in the selected node over time / frequency (only for more steps)."""
        if not self.has_history:
            return None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure
        except Exception as error:
            print(f"Graph is not available (matplotlib with Qt backend is needed): {error}", file=sys.stderr)
            return None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        top = QtWidgets.QHBoxLayout()
        self.graph_label = QtWidgets.QLabel()
        top.addWidget(self.graph_label, stretch=1)
        self.hold_check = QtWidgets.QCheckBox("Hold curves")
        self.hold_check.setToolTip("Keep the curves of previously selected nodes / results (comparison)")
        self.hold_check.toggled.connect(lambda on: self._update_graph())
        self.logy_check = QtWidgets.QCheckBox("Log Y")
        self.logy_check.toggled.connect(lambda on: self._update_graph())
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda checked=False: self._clear_curves())
        export = QtWidgets.QPushButton("Export CSV...")
        export.clicked.connect(lambda checked=False: self._on_export_csv())
        for w in (self.hold_check, self.logy_check, clear, export):
            top.addWidget(w)
        self.figure = Figure(figsize=(8, 2.6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.graph_toolbar = NavigationToolbar2QT(self.canvas, widget)
        self.canvas.mpl_connect("button_press_event", self._on_graph_click)
        layout.addLayout(top)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.graph_toolbar)
        return widget

    def _history(self, name, index):
        key = (name, index, self.steps[self.step]["subcase"], self.steps[self.step]["kind"], self.cmode)
        if key not in self._history_cache:
            self._history_cache[key] = self.h5.history(name, index, self.step, self.cmode)
        return self._history_cache[key]

    def _clear_curves(self):
        self.curves = []
        self._update_graph()

    def _update_graph(self):
        if self.graph_widget is None:
            return
        name, location = self.results[self.idx]
        current = None
        if self.selected is not None:
            node, cell = self.selected
            index = node if location == "point" else cell
            if index is not None:
                history = self._history(name, index)
                if history is not None:
                    steps, x, y = history
                    where = self._where_text(node, cell, location)
                    label = f"{where}: {name}" + (f" [{self.cmode}]" if self.h5.is_complex(name) else "")
                    current = dict(key=(name, index, self.cmode), label=label, steps=steps, x=x, y=y)
        if current is not None:
            if self.hold_check.isChecked():
                if current["key"] not in [c["key"] for c in self.curves]:
                    self.curves.append(current)
            else:
                self.curves = [current]
        elif not self.hold_check.isChecked():
            self.curves = []

        ax = self.axes
        ax.clear()
        for curve in self.curves:
            ax.plot(curve["x"], curve["y"], "-", marker=".", markersize=3, linewidth=1.2, label=curve["label"])
            if self.step in curve["steps"]:                     # marker = current step
                i = int(np.flatnonzero(curve["steps"] == self.step)[0])
                ax.plot(curve["x"][i], curve["y"][i], "o", color="red", markersize=6)
        if self.steps:
            ax.axvline(self.steps[self.step]["x"], color="red", linewidth=0.8, linestyle="--")
        ax.set_xlabel(self.h5.step_axis(self.step))
        ax.set_ylabel(name + (f" [{self.cmode}]" if self.h5.is_complex(name) else ""), fontsize=8)
        if self.logy_check.isChecked() and any(np.nanmax(c["y"]) > 0 for c in self.curves if c["y"].size):
            ax.set_yscale("log", nonpositive="mask")
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)
        if len(self.curves) > 1:
            ax.legend(fontsize=7)
        if self.curves:
            self.graph_label.setText(f"<b>{self.curves[-1]['label']}</b>  (click into the graph = show that step)")
        elif self.selected is None:
            self.graph_label.setText("Select a node (double-click in 3D, key I or node ID) to see the course of "
                                     "the current result over the steps.")
        else:
            self.graph_label.setText(f"No course of {name} in the selected node / element.")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_graph_click(self, event):
        if event.inaxes is not self.axes or event.button != 1 or event.xdata is None:
            return
        if str(getattr(self.graph_toolbar, "mode", "")):        # zoom / pan of the graph is active
            return
        current = self.steps[self.step]
        candidates = [i for i, s in enumerate(self.steps) if s["subcase"] == current["subcase"]
                      and s["kind"] == current["kind"]]
        x = np.array([self.steps[i]["x"] for i in candidates])
        self.set_step(candidates[int(np.argmin(np.abs(x - event.xdata)))])

    def _on_export_csv(self):
        if not self.curves:
            return
        default = self.h5.file_path.with_name(self.h5.file_path.stem + "_history.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Export curves", str(default),
                                                        "CSV file (*.csv)")
        if path:
            self.export_curves(path)

    def export_curves(self, path):
        """Saves the curves of the graph to a CSV file (x, y of every curve)."""
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(sum([[self.h5.step_axis(self.step), c["label"]] for c in self.curves], []))
            for i in range(max(len(c["x"]) for c in self.curves)):
                row = []
                for c in self.curves:
                    row += [f"{c['x'][i]:.8g}", f"{c['y'][i]:.8g}"] if i < len(c["x"]) else ["", ""]
                writer.writerow(row)

    # ================================================================================================= geometry
    @staticmethod
    def _surface(dataset, pass_ids=False):
        surface_filter = vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(dataset)
        if pass_ids:
            surface_filter.PassThroughPointIdsOn()                # "vtkOriginalPointIds"
        surface_filter.Update()
        return pv.wrap(surface_filter.GetOutput())

    @staticmethod
    def _feature_edges(surface):
        edges = vtkFeatureEdges()
        edges.SetInputData(surface)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOn()
        edges.SetFeatureAngle(30.0)
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ColoringOff()
        edges.Update()
        return pv.wrap(edges.GetOutput())

    def _build_surface(self):
        """Outer surface + outline of the whole model (geometry only, results are mapped onto it)."""
        self.surface = self._surface(self.grid, pass_ids=True)
        self.surface_points0 = self.surface.points.copy()
        self.surface_ids = np.asarray(self.surface.point_data["vtkOriginalPointIds"])
        self.surface_cells = np.asarray(self.surface.cell_data["_cell_index"])
        self.outline = self._feature_edges(self.surface)        # model outline shown with sections
        self.outline_points0 = self.outline.points.copy()
        self.outline_ids = np.asarray(self.outline.point_data["vtkOriginalPointIds"])
        self.outline_actor = self.pl.add_mesh(self.outline, color="gray", opacity=0.5, line_width=1,
                                              name="outline", pickable=False, reset_camera=False, render=False)
        self.outline_actor.SetVisibility(False)

    def _load_step(self):
        """Deformation vectors of the current step."""
        vectors = self.h5.get_deformation(self.step, self.phase)
        if vectors is None:
            vectors = np.zeros((self.grid.n_points, 3))
        self.U = np.nan_to_num(vectors)
        self.u_max = float(np.linalg.norm(self.U, axis=1).max())

    def _update_mesh(self):
        """Mesh to display (outer surface / surface of the cut model / section plane only) with the values of
        the current result ("_value") and the deformation vectors ("_U")."""
        name, location = self.results[self.idx]
        values = self._values()
        if not self.section:
            self.mesh = self.surface
            self.mesh_points0 = self.surface_points0
            self.mesh_U = self.U[self.surface_ids]
            self.mesh.point_data.pop("_value", None)
            self.mesh.cell_data.pop("_value", None)
            if location == "point":
                self.mesh.point_data["_value"] = values[self.surface_ids]
            else:
                self.mesh.cell_data["_value"] = values[self.surface_cells]
            return

        work = pv.UnstructuredGrid()
        work.ShallowCopy(self.grid)
        work.point_data["_U"] = self.U
        if location == "point":
            work.point_data["_value"] = values
        else:
            work.cell_data["_value"] = values
        plane = vtkPlane()
        plane.SetNormal(*self.normal)
        plane.SetOrigin(*self.origin)
        if self.slice_only:
            cutter = vtkCutter()
            cutter.SetInputData(work)
            cutter.SetCutFunction(plane)
            cutter.Update()
            self.mesh = pv.wrap(cutter.GetOutput())
        else:
            clipper = vtkTableBasedClipDataSet()
            clipper.SetInputData(work)
            clipper.SetClipFunction(plane)
            clipper.InsideOutOn()                       # keep the part behind the normal
            clipper.Update()
            self.mesh = self._surface(clipper.GetOutput())
        self.mesh_points0 = self.mesh.points.copy()
        self.mesh_U = np.asarray(self.mesh.point_data["_U"]) if self.mesh.n_points else np.zeros((0, 3))

    # ================================================================================================= 3D scene
    def _build_scene(self):
        self._update_mesh()
        self._render_model()

        # section plane widget - hidden until the section is switched on
        self.plane_widget = self.pl.add_plane_widget(
            callback=self._on_plane_widget, normal=self.normal, origin=self.origin, bounds=self.grid.bounds,
            factor=1.05, color="black", implicit=True, normal_rotation=True, origin_translation=True,
            outline_translation=False, interaction_event="end", test_callback=False)
        self.plane_widget.GetOutlineProperty().SetOpacity(0.25)
        self.plane_widget.Off()

        self.pl.add_key_event("Right", lambda: self.set_result(self.idx + 1))
        self.pl.add_key_event("Left", lambda: self.set_result(self.idx - 1))
        self.pl.add_key_event("Prior", lambda: self.set_step(self.step - 1))
        self.pl.add_key_event("Next", lambda: self.set_step(self.step + 1))
        self.pl.add_key_event("i", self._on_probe)
        self.pl.add_key_event("I", self._on_probe)
        for key, normal in (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))):
            for k in (key, key.upper()):
                self.pl.add_key_event(k, lambda n=normal: self.set_section_normal(n))
        self.pl.add_key_event("n", self.flip_section)
        self.pl.add_key_event("N", self.flip_section)
        # double-click = select the node under the mouse cursor
        self.pl.iren.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button, 1.0)

        self.pl.add_axes()
        self.pl.view_isometric()
        self._update_selection()

    # ================================================================================================= results
    def _values(self, idx=None):
        """Values of a result (all nodes / elements) in the current step and complex part."""
        name, _ = self.results[self.idx if idx is None else idx]
        return np.asarray(self.h5.values(name, self.step, self.cmode), dtype=float)

    def _shown_values(self):
        """Finite values of the current result on the displayed mesh (outer surface / cut model / slice)."""
        name, location = self.results[self.idx]
        if self.mesh.n_cells == 0:
            return np.array([])
        data = self.mesh.point_data if location == "point" else self.mesh.cell_data
        values = np.asarray(data["_value"], dtype=float)
        return values[np.isfinite(values)]

    def set_result(self, idx):
        """Shows result with given index (in self.results)."""
        idx = int(np.clip(idx, 0, len(self.results) - 1))
        if idx != self.idx:
            self.idx = idx
            self._refresh()
        self.pl.render()

    def set_step(self, step):
        """Shows result step `step` (index in self.steps: time step, frequency, mode)."""
        if not self.steps:
            return
        step = int(np.clip(step, 0, len(self.steps) - 1))
        if step != self.step:
            self.step = step
            self._load_step()
            self._refresh()
            QtCore.QTimer.singleShot(0, self._update_tree_ranges)
        self.pl.render()

    def set_complex_mode(self, mode):
        """Shown part of complex results: "amplitude", "phase", "real", "imag"."""
        if mode != self.cmode:
            self.cmode = mode
            self._refresh()
            self._update_tree_ranges()
        self.pl.render()

    def set_phase(self, phase):
        """Phase angle [deg] of the deformed shape of complex results."""
        self.phase = float(phase)
        self._load_step()
        self._update_mesh()
        self._apply_deformation()
        self._update_selection()
        self.pl.render()

    def _refresh(self):
        """Redraw after a change of the result / step / complex part."""
        self._update_mesh()
        self._render_model()
        self._update_selection()
        self._sync_controls()

    def _render_model(self):
        """(Re)creates the actor of the displayed mesh colored by the current result."""
        name, location = self.results[self.idx]
        values = self._values()
        finite = self._shown_values() if self.local_range else values[np.isfinite(values)]
        clim = [float(finite.min()), float(finite.max())] if finite.size else [0.0, 1.0]
        if clim[0] == clim[1]:
            clim = [clim[0] - 0.5, clim[1] + 0.5]

        if self.actor is not None:
            self.pl.remove_actor(self.actor, render=False)
            try:
                self.pl.remove_scalar_bar(self._bar_title, render=False)
            except (KeyError, ValueError, AttributeError):
                pass
            self.actor = None

        if self.mesh.n_cells > 0:                       # section outside of the model -> nothing to show
            title = name + (f" [{self.cmode}]" if self.h5.is_complex(name) else "")
            self._bar_title = title + "\n"              # new line = space between the title and the top label
            # IDs, codes, ... -> one color per integer value (+ text labels of the values if available)
            is_id = name in getattr(self.h5, "categorical_results", {"Property ID"})
            annotations = getattr(self.h5, "result_annotations", {}).get(name)
            n_colors = int(min(256, clim[1] - clim[0] + 1)) if is_id else 256
            n_labels = (0 if annotations else min(n_colors, 11)) if is_id else 11
            self.actor = self.pl.add_mesh(
                self.mesh, scalars="_value", preference=location, cmap=self.cmap, clim=clim, n_colors=n_colors,
                show_edges=self.show_edges, edge_color=(0.25, 0.25, 0.25), nan_color="lightgray",
                ambient=0.35, diffuse=0.65, specular=0.0, reset_camera=False, render=False,
                annotations=annotations,
                scalar_bar_args=dict(title=self._bar_title, vertical=True, position_x=0.86, position_y=0.2,
                                     width=0.06, height=0.65, fmt="%.0f" if is_id else "%.3e",
                                     n_labels=n_labels, title_font_size=14, label_font_size=12, color="black"))
        self._apply_deformation()
        self._update_info()

    # ================================================================================================= display
    def _on_edges(self, state):
        self.show_edges = bool(state)
        if self.actor is not None:
            self.actor.prop.show_edges = self.show_edges
        self.pl.render()

    def _on_local_range(self, state):
        self.local_range = bool(state)
        self._render_model()
        self.pl.render()

    def _on_deform_slider(self, value):
        """Deformation in % of the model size for the current step -> scale factor (kept for other steps)."""
        self.deform_pct = float(value)
        self.scale = self.deform_pct / 100.0 * self.model_size / self.u_max if self.u_max > 0 else 0.0
        self._apply_deformation()
        self._update_info()
        self._update_selection()
        self._sync_controls()
        self.pl.render()

    def _apply_deformation(self):
        if self.mesh.n_points > 0:
            self.mesh.points = self.mesh_points0 + self.scale * self.mesh_U
        self.outline.points = self.outline_points0 + self.scale * self.U[self.outline_ids]

    # ================================================================================================= section
    def _model_range(self, normal):
        """Min / max distance of the model bounding box corners along the normal."""
        b = self.grid.bounds
        corners = np.array([[x, y, z] for x in b[0:2] for y in b[2:4] for z in b[4:6]])
        d = corners @ normal
        return d.min(), d.max()

    def _section_fraction(self):
        d_min, d_max = self._model_range(self.normal)
        return float(np.clip((self.origin @ self.normal - d_min) / (d_max - d_min) * 100.0, 0.0, 100.0))

    def _update_section(self):
        """Rebuilds the displayed mesh after any change of the section and renders."""
        self.plane_widget.SetNormal(*self.normal)
        self.plane_widget.SetOrigin(*self.origin)
        self.plane_widget.UpdatePlacement()
        self.plane_widget.SetEnabled(self.section and self.plane_widget_on)
        self.outline_actor.SetVisibility(self.section)
        self._update_mesh()
        self._render_model()
        self._update_selection()
        self._sync_controls()
        self.pl.render()

    def set_section(self, on=True, normal=None, origin=None, slice_only=None):
        """Switches the section on / off, optionally sets the plane (normal points to the removed part)."""
        self.section = bool(on)
        if normal is not None:
            normal = np.asarray(normal, dtype=float)
            self.normal = normal / np.linalg.norm(normal)
        if origin is not None:
            self.origin = np.asarray(origin, dtype=float)
        if slice_only is not None:
            self.slice_only = bool(slice_only)
        self._update_section()

    def set_section_normal(self, normal):
        self.set_section(True, normal=normal)

    def set_section_position(self, percent):
        """Moves the section plane along its normal, 0 - 100 % of the model."""
        d_min, d_max = self._model_range(self.normal)
        d = d_min + float(np.clip(percent, 0.0, 100.0)) / 100.0 * (d_max - d_min)
        self.origin = self.origin + (d - self.origin @ self.normal) * self.normal
        self.set_section(self.section)

    def flip_section(self):
        if self.section:
            self.set_section(True, normal=-self.normal)

    def _on_plane_check(self, state):
        self.plane_widget_on = bool(state)
        self.plane_widget.SetEnabled(self.section and self.plane_widget_on)
        self.pl.render()

    def _on_plane_widget(self, normal, origin):
        b = np.array(self.grid.bounds).reshape(3, 2)
        self.set_section(True, normal=normal, origin=np.clip(origin, b[:, 0], b[:, 1]))

    def _section_text(self):
        axis = np.flatnonzero(np.isclose(np.abs(self.normal), 1.0))
        if axis.size:
            k = axis[0]
            plane = f"{'XYZ'[k]} = {self.origin[k]:.5g}, removed {'+-'[int(self.normal[k] < 0)]}{'XYZ'[k]} side"
        else:
            n, o = self.normal, self.origin
            plane = f"normal ({n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}), origin ({o[0]:.3g}, {o[1]:.3g}, {o[2]:.3g})"
        return f"{plane}  ({self._section_fraction():.1f} %)"

    # ================================================================================================= selection
    def _on_left_button(self, interactor, event):
        if interactor.GetRepeatCount() > 0:                     # double-click
            self.probe_at(*interactor.GetEventPosition())

    def _on_probe(self):
        x, y = self.pl.iren.interactor.GetEventPosition()
        self.probe_at(x, y)

    def probe_at(self, x, y):
        """Selects the node (and element) at window position x, y (pixels from the lower left corner)."""
        if self.actor is None:
            return
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()                         # pick only the model (not labels etc.)
        picker.AddPickList(self.actor)
        picker.Pick(x, y, 0, self.pl.renderer)
        if picker.GetDataSet() is None or picker.GetCellId() < 0:
            return
        cell = int(pv.wrap(picker.GetDataSet()).cell_data["_cell_index"][picker.GetCellId()])
        # nearest node of the picked element (in the current, possibly deformed shape)
        nodes = np.asarray(self.grid.get_cell(cell).point_ids)
        positions = self.grid.points[nodes] + self.scale * self.U[nodes]
        distance = np.linalg.norm(positions - np.array(picker.GetPickPosition()), axis=1)
        self.select(int(nodes[np.argmin(distance)]), cell)

    def select_node_id(self, node_id):
        """Selects node by its ID, returns False for an unknown ID."""
        index = int(self.h5.node_index([int(node_id)])[0])
        if index < 0:
            return False
        self.select(index, self._cell_of_node(index))
        return True

    def _on_node_edit(self):
        text = self.node_edit.text().strip()
        try:
            found = self.select_node_id(int(text))
        except ValueError:
            found = False
        if not found:
            self.selection_label.setText(f"<span style='color:red'>Node '{text}' not found</span>")

    def _cell_of_node(self, node):
        """Some element using the node (for element results of a node selected by its ID, also midside nodes)."""
        full = self.h5.to_pyvista()                             # same cells as self.grid, incl. midside nodes
        positions = np.flatnonzero(np.asarray(full.cell_connectivity) == node)
        if positions.size == 0:
            return None
        return int(np.searchsorted(np.asarray(full.offset), positions[0], side="right") - 1)

    def select(self, node, cell=None):
        """Selects node (point index) and element (cell index) - shows its values and the graph; None = clear."""
        self.selected = None if node is None else (int(node), None if cell is None else int(cell))
        self._update_selection()
        self._update_info()
        self.pl.render()

    def _where_text(self, node, cell, location):
        if location == "point" or cell is None:
            return f"node {self.h5.node_ids[node]}"
        return f"element {self.h5.element_ids[cell]}"

    def _update_selection(self):
        """Marker in 3D, table of values of the selected node and the graph."""
        if self.selected is None:
            self.pl.remove_actor("probe", render=False)
            self.selection_label.setText("No node selected.")
            self.selection_table.clear()
            self.selection_table.hide()
            self._update_graph()
            return
        node, cell = self.selected
        name, location = self.results[self.idx]
        if not self.node_edit.hasFocus():
            self.node_edit.setText(str(self.h5.node_ids[node]))
        p = self.grid.points[node]
        text = f"<b>Node {self.h5.node_ids[node]}</b>  ({p[0]:.5g}, {p[1]:.5g}, {p[2]:.5g})"
        if cell is not None:
            text += (f"<br>element {self.h5.element_ids[cell]} ({self.h5.element_types[cell]}, "
                     f"PID {self.h5.property_ids[cell]})")
        if self.steps:
            text += f"<br>{self.steps[self.step]['label']}"
        self.selection_label.setText(text)

        # values of the results of the current group (e.g. Displacement magnitude, X, Y, Z)
        group = self.h5.result_meta.get(name, ("", ""))[0]
        self.selection_table.clear()
        bold = QtGui.QFont()
        bold.setBold(True)
        current_value = np.nan
        for i, (other, other_location) in enumerate(self.results):
            if self.h5.result_meta.get(other, ("", ""))[0] != group:
                continue
            index = node if other_location == "point" else cell
            if index is None:
                continue
            value = self.h5.values(other, self.step, self.cmode)[index]
            shown = _fmt(value)
            if self.h5.is_complex(other) and self.cmode != "phase":
                amplitude = self.h5.values(other, self.step, "amplitude")[index]
                phase = self.h5.values(other, self.step, "phase")[index]
                if np.isfinite(phase):
                    shown = f"{_fmt(value)}   ({_fmt(amplitude)} / {phase:.1f} deg)"
            item = QtWidgets.QTreeWidgetItem(self.selection_table, [self.h5.result_meta[other][1], shown])
            item.setToolTip(0, other)
            if i == self.idx:
                item.setFont(0, bold)
                item.setFont(1, bold)
                current_value = value
        self.selection_table.resizeColumnToContents(0)
        self.selection_table.setVisible(self.selection_table.topLevelItemCount() > 0)

        # marker + label in 3D (current, possibly deformed position)
        position = p + self.scale * self.U[node]
        where = self._where_text(node, cell, location)
        self.pl.add_point_labels(position[np.newaxis, :], [f"{where}\n{name} = {_fmt(current_value)}"],
                                 name="probe", font_size=12, point_size=12, point_color="magenta",
                                 text_color="black", shape_color="white", shape_opacity=0.8, always_visible=True,
                                 render_points_as_spheres=True, render=False)
        self._update_graph()

    # ================================================================================================= info text
    def _update_info(self):
        name, location = self.results[self.idx]
        values = self._values()
        lines = [f"{self.h5.file_path.name}", self.steps[self.step]["label"] if self.steps else ""]
        title = name + (f" [{self.cmode}]" if self.h5.is_complex(name) else "")
        if name in getattr(self.h5, "categorical_results", {"Property ID"}):
            labels = getattr(self.h5, "result_annotations", {}).get(name, {})
            unique = np.unique(values[np.isfinite(values)]).astype(int)
            shown = ", ".join(str(labels.get(v, v)) for v in unique[:12]) + (", ..." if len(unique) > 12 else "")
            lines.append(title)
            lines.append(f"{len(unique)} values: {shown}")
        elif np.isfinite(values).any():
            i_min, i_max = np.nanargmin(values), np.nanargmax(values)
            ids = self.h5.node_ids if location == "point" else self.h5.element_ids
            what = "node" if location == "point" else "element"
            lines.append(title)
            lines.append(f"min = {values[i_min]:.4e}  ({what} {ids[i_min]})")
            lines.append(f"max = {values[i_max]:.4e}  ({what} {ids[i_max]})")
        else:
            lines.append(f"{title}: no values in this step")
        if self.section:
            lines.append(("Slice: " if self.slice_only else "Section: ") + self._section_text())
        if self.section or self.local_range:
            shown = self._shown_values()
            if shown.size:
                lines.append(f"shown part: min = {shown.min():.4e}, max = {shown.max():.4e}"
                             + ("  (= color range)" if self.local_range else ""))
            elif self.mesh.n_cells == 0:
                lines.append("shown part: nothing (plane outside of the model)")
        if self.scale > 0:
            lines.append(f"Deformation scale = {self.scale:.4g}x"
                         + (f", phase {self.phase:.0f} deg" if self.has_complex else ""))
        if self.selected is not None:
            node, cell = self.selected
            index = node if location == "point" else cell
            if index is not None:
                lines.append(f"Selected {self._where_text(node, cell, location)}: {_fmt(values[index])}")
        self.pl.add_text("\n".join(line for line in lines if line), position="upper_left", font_size=10,
                         color="black", name="info")

    # ================================================================================================= run
    def show(self):
        """Opens the window and runs the Qt event loop (returns after the window is closed)."""
        self.window.show()
        self.pl.render()
        run = getattr(self.app, "exec", None) or getattr(self.app, "exec_")
        run()

    def screenshot(self, file_path):
        """Saves the 3D view to a PNG file."""
        self.pl.render()
        self.pl.screenshot(str(file_path))
