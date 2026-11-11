"""
Interactive 3D viewer of MSC Nastran H5 results (read by file_processing.read_Nastran_files.ParseH5File).

Window (visualization/viewer_window.py): menu File - Open (Ctrl+O), Open recent, Reload (F5), Save / restore camera
view; a file can be dragged & dropped into the window; files are loaded in the background (progress in the status
bar), so the window does not freeze.
    left   - tab "Results": result step (time / frequency / mode) with amplitude / phase / real / imaginary part of
             complex results, table of all results (click / arrow keys = show the result), display (deformation,
             animation), section, selected node (double-click in 3D or node ID) with the values of the current group
           - tab "Random" (frequency response): random vibration - input acceleration PSD, response PSD in the
             selected node, RMS / 3 sigma maps of the whole model (file_processing/random_response.py)
           - tab "Modes" (eigenvalue summary in the file): natural frequencies, click = mode shape / nearest frequency
    right  - 3D view (pyvista) and, if the results have more steps, a graph of the current result in the selected
             node over time / frequency (click into the graph = show that step) with the natural frequencies as
             vertical lines, or the response PSD of the random vibration

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
import sys
import time
import traceback
from collections import Counter

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter, vtkFeatureEdges
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkRenderingCore import vtkCellPicker

from visualization.qt_base import (HORIZONTAL, ITEM_ENABLED, TEXT_SELECTABLE, USER_ROLE, VERTICAL,  # noqa: F401
                                   QtCore, QtGui, QtInteractor, QtWidgets, TaskRunner, qt_app, run_app, safe_call)

COMPLEX_MODES = (("amplitude", "Amplitude"), ("phase", "Phase [deg]"), ("real", "Real part"),
                 ("imag", "Imaginary part"))


def _fmt(value):
    return "-" if value is None or not np.isfinite(value) else f"{value:.4e}"


class H5ResultViewer:
    def __init__(self, h5, cmap="jet", show_edges=True, window_size=(1650, 980), window=None, prepared=None):
        """
        Args:
            h5 (ParseH5File / NastranMesh): loaded Nastran results (or model)
            cmap (str): matplotlib colormap name
            show_edges (bool): show element edges at start
            window_size (tuple): window size in pixels (if a new window is created)
            window (ViewerWindow): window to show the viewer in (None = new window)
            prepared (dict): result of H5ResultViewer.prepare(h5) (computed in the background by the window)
        """
        self.app = qt_app()

        self.h5 = h5
        h5.to_pyvista()                                 # fills h5.results
        self.results = h5.results                       # [(name, "point" / "cell")] - shared list (new results)
        self.steps = list(getattr(h5, "steps", []))     # result steps (time, frequency, modes)
        self.has_complex = any(h5.is_complex(name) for name, _ in self.results)
        per_series = Counter(self._series(i) for i in range(len(self.steps)))
        self.has_history = any(n > 1 for n in per_series.values())

        # linear copy of the mesh (same points, cells and data arrays) + outer surface (prepared in the background)
        self._prepared = prepared or self.prepare(h5)
        self.grid = self._prepared["grid"]

        self.cmap = cmap
        self.show_edges = show_edges
        self.local_range = False                        # color range from the shown part / the whole model
        self.idx = 0                                    # index of the displayed result
        self.step = 0                                   # index of the displayed result step
        self.cmode = "amplitude"                        # shown part of complex results
        self.phase = 0.0                                # phase angle of the deformed shape of complex results
        self.anim_phase = 0.0                           # phase of the animation [rad]
        self.deform_pct = 0.0                           # max deformation in % of the model size (at slider change)
        self.scale = 0.0                                # deformation scale factor (kept when the step changes)
        self.selected = None                            # (node index, cell index or None)
        self.curves = []                                # curves in the graph
        self.highlight_mode = None                      # mode number highlighted in the graph (tab Modes)
        self._history_cache = {}
        self.model_size = self.grid.length              # diagonal of the bounding box
        self.random_tab = None                          # tab "Random" (frequency response)
        self.modes_tab = None                           # tab "Modes" (eigenvalue summary)
        self._disposed = False

        self.section = False                            # section on / off
        self.slice_only = False                         # True = only the section plane, False = cut model
        self.plane_widget_on = True                     # interactive plane widget in the 3D view
        self.normal = np.array([0.0, 0.0, 1.0])         # section plane, normal points to the removed part
        self.origin = np.array(self.grid.center, dtype=float)

        if window is None:
            from visualization.viewer_window import ViewerWindow
            window = ViewerWindow(window_size=window_size)
        self.window = window
        self.tasks = TaskRunner()                       # background work (min / max of results, random RMS)
        self.tasks.on_progress = getattr(window, "show_task_progress", None)
        self.tasks.on_finished = getattr(window, "task_finished", None)
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.setInterval(40)
        self.anim_timer.timeout.connect(lambda: safe_call(self._animation_frame))
        self._anim_t0 = 0.0

        # layout: panel | (3D view / graph)
        self.pl = QtInteractor()
        self.pl.set_background("white")
        splitter = QtWidgets.QSplitter(HORIZONTAL)
        splitter.addWidget(self._build_side())
        right = QtWidgets.QSplitter(VERTICAL)
        right.addWidget(self.pl.interactor)
        self.graph_widget = self._build_graph()
        if self.graph_widget is not None:
            right.addWidget(self.graph_widget)
            height = max(window.height(), 600)
            right.setSizes([height - 330, 300])
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, max(window.width(), 1000) - 460])
        self.widget = splitter

        self.actor = None
        self._bar_title = None
        self._build_surface()
        self._load_step()
        self._build_scene()
        self._update_tree_ranges()
        self._sync_controls()
        self._prepared = None
        window.set_viewer(self)

    @classmethod
    def prepare(cls, h5, progress=None):
        """
        Preparation of the 3D view which may take long for big models - it can run in a background thread
        (no Qt / rendering): linear copy of the mesh, outer surface, outline, values of the first result.
        """
        report = progress or (lambda fraction, text: None)
        report(0.85, "preparing the 3D view")
        grid = pv.UnstructuredGrid()
        grid.ShallowCopy(h5.to_pyvista(linear=True))
        grid.cell_data["_cell_index"] = np.arange(grid.n_cells)
        surface = cls._surface(grid, pass_ids=True)
        outline = cls._feature_edges(surface)
        report(0.95, "reading the first result")
        if h5.results:
            h5.values(h5.results[0][0], 0, "amplitude")         # -> cache
        h5.deformation_vectors(0)
        return dict(grid=grid, surface=surface, outline=outline)

    def title(self):
        """Title of the window."""
        return f"Nastran results - {self.h5.file_path.name}"

    def _series(self, step):
        """Course (subcase, kind, ...) the step belongs to - steps of one course make a history graph."""
        s = self.steps[step]
        return s.get("series", (s["subcase"], s["kind"]))

    def _series_steps(self, step=None):
        series = self._series(self.step if step is None else step)
        return [i for i in range(len(self.steps)) if self._series(i) == series]

    def _status(self, text):
        if hasattr(self.window, "status"):
            self.window.status(text)

    # ================================================================================================= Qt panel
    def _build_side(self):
        """Left side: panel "Results" (+ tabs "Random" / "Modes" if the results allow it)."""
        panel = self._build_panel()
        from visualization.dynamics_tabs import ModesTab, RandomTab

        tabs = []
        if RandomTab.available(self.h5):
            self.random_tab = RandomTab(self)
            tabs.append((self.random_tab.widget, "Random"))
        if ModesTab.available(self.h5):
            self.modes_tab = ModesTab(self)
            tabs.append((self.modes_tab.widget, "Modes"))
        if not tabs:
            return panel
        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.addTab(panel, "Results")
        for widget, label in tabs:
            self.side_tabs.addTab(widget, label)
        return self.side_tabs

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
        self._rebuild_tree()
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
        self.animate_button = QtWidgets.QPushButton("Animate")
        self.animate_button.setCheckable(True)
        self.animate_button.setToolTip("Animation of the deformed shape: modes and real results U * cos(phi), "
                                       "frequency response Re(U * exp(i phi)), phi = 0 - 360 deg")
        self.animate_button.toggled.connect(lambda on: self.set_animation(on))
        self.period_spin = QtWidgets.QDoubleSpinBox()
        self.period_spin.setRange(0.3, 20.0)
        self.period_spin.setSingleStep(0.5)
        self.period_spin.setValue(2.0)
        self.period_spin.setSuffix(" s / cycle")
        self.period_spin.setToolTip("Duration of one cycle of the animation")
        grid.addWidget(self.edges_check, 0, 0)
        grid.addWidget(self.local_check, 0, 1)
        grid.addWidget(deform_caption, 1, 0)
        grid.addWidget(self.deform_slider, 1, 1)
        grid.addWidget(self.deform_label, 2, 0, 1, 2)
        grid.addWidget(self.animate_button, 3, 0)
        grid.addWidget(self.period_spin, 3, 1)
        if self.h5.deformation_vectors(0) is None:
            for widget in (deform_caption, self.deform_slider, self.deform_label, self.animate_button,
                           self.period_spin):
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
        for text, callback, tip in (
                ("Reset view", lambda: self.pl.reset_camera(), "Whole model in the view (key R)"),
                ("Isometric", lambda: self.pl.view_isometric(), "Isometric view (key V)"),
                ("Save view", lambda: self.window.save_view(),
                 "Save the camera view for this file - restored when the file is opened again (File menu)"),
                ("Screenshot...", self._on_screenshot, "Save the 3D view as PNG")):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda checked=False, c=callback: safe_call(c))
            buttons.addWidget(button)
        layout.addLayout(buttons)

        help_label = QtWidgets.QLabel("3D view: left mouse = rotate, middle / Shift+left = pan, wheel = zoom, "
                                      "double-click = select node; keys I = select node, Left/Right = result, "
                                      "PageUp/PageDown = step, X/Y/Z = section, N = flip, R = reset view, "
                                      "V = isometric. Another file: File > Open or drag & drop.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray")
        layout.addWidget(help_label)
        return panel

    def _rebuild_tree(self):
        """(Re)fills the table of results (groups -> results); min / max are filled in the background."""
        self.tree.blockSignals(True)
        old = {i: (item.text(1), item.text(2)) for i, item in self._tree_items.items()}
        self.tree.clear()
        self._tree_items = {}
        parents = {}
        bold = QtGui.QFont()
        bold.setBold(True)
        for i, (name, location) in enumerate(self.results):
            group, item_label = self.h5.result_meta.get(name, ("Other", name))
            if group not in parents:
                parents[group] = self._group_item(self.tree, group, bold)
            item = QtWidgets.QTreeWidgetItem(parents[group], [item_label, *old.get(i, ("", ""))])
            item.setData(0, USER_ROLE, i)
            kind = "nodal" if location == "point" else "element"
            item.setToolTip(0, f"{name}  ({kind} values" + (", complex)" if self.h5.is_complex(name) else ")"))
            self._tree_items[i] = item
        self.tree.expandAll()
        if self.idx in self._tree_items:
            self.tree.setCurrentItem(self._tree_items[self.idx])
        self.tree.blockSignals(False)

    def results_changed(self, select=None):
        """Results were added or changed (e.g. random RMS maps): table of results, 3D view, min / max."""
        if select is not None:
            names = [name for name, _ in self.results]
            if select in names:
                self.idx = names.index(select)
        self._rebuild_tree()
        self._history_cache.clear()
        self._refresh()
        self._update_tree_ranges()
        self.pl.render()

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
        """Min / max of all results in the current step (and complex part) - computed in the background."""
        if self._disposed:
            return
        h5, results, step, cmode = self.h5, list(self.results), self.step, self.cmode

        def work(task):
            ranges = []
            for i, (name, _) in enumerate(results):
                task.check()
                values = np.asarray(h5.values(name, step, cmode), dtype=float)
                finite = values[np.isfinite(values)]
                ranges.append((i, float(finite.min()), float(finite.max())) if finite.size else (i, None, None))
            return ranges

        self.tasks.start("ranges", work, done=self._set_tree_ranges)

    def _set_tree_ranges(self, ranges):
        if self._disposed:
            return
        for i, low, high in ranges:
            item = self._tree_items.get(i)
            if item is not None:
                item.setText(1, "-" if low is None else f"{low:.4g}")
                item.setText(2, "-" if high is None else f"{high:.4g}")
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
        if self.idx in self._tree_items:
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
        self.psd_check = QtWidgets.QCheckBox("Response PSD")
        self.psd_check.setToolTip("Random vibration: response PSD of the current result in the selected node "
                                  "(input PSD in the tab Random) instead of the frequency response")
        self.psd_check.toggled.connect(lambda on: self._update_graph())
        self.psd_check.setVisible(self.random_tab is not None)
        self.modes_check = QtWidgets.QCheckBox("Natural frequencies")
        self.modes_check.setToolTip("Natural frequencies (eigenvalue summary) as vertical lines")
        self.modes_check.setChecked(True)
        self.modes_check.toggled.connect(lambda on: self._update_graph())
        self.modes_check.setVisible(bool(getattr(self.h5, "modes", None)))
        self.hold_check = QtWidgets.QCheckBox("Hold curves")
        self.hold_check.setToolTip("Keep the curves of previously selected nodes / results (comparison)")
        self.hold_check.toggled.connect(lambda on: self._update_graph())
        self.logy_check = QtWidgets.QCheckBox("Log Y")
        self.logy_check.toggled.connect(lambda on: self._update_graph())
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda checked=False: self._clear_curves())
        export = QtWidgets.QPushButton("Export CSV...")
        export.clicked.connect(lambda checked=False: self._on_export_csv())
        for w in (self.psd_check, self.modes_check, self.hold_check, self.logy_check, clear, export):
            top.addWidget(w)
        self.figure = Figure(figsize=(8, 2.6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._twin = None
        self.graph_toolbar = NavigationToolbar2QT(self.canvas, widget)
        self.canvas.mpl_connect("button_press_event", self._on_graph_click)
        layout.addLayout(top)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.graph_toolbar)
        return widget

    def _history(self, name, index):
        key = (name, index, self._series(self.step), self.cmode)
        if key not in self._history_cache:
            self._history_cache[key] = self.h5.history(name, index, self.step, self.cmode)
        return self._history_cache[key]

    def _clear_curves(self):
        self.curves = []
        self._update_graph()

    def _graph_kind(self):
        """"psd" = response PSD of a random vibration, "history" = result over time / frequency / modes."""
        rt = self.random_tab
        if rt is None or getattr(self, "graph_widget", None) is None or not self.results:
            return "history"
        name = self.results[self.idx][0]
        if rt.is_random_result(name) or (self.psd_check.isChecked() and rt.supports(name)):
            return "psd"
        return "history"

    def _frequency_axis(self, kind):
        return kind == "psd" or (bool(self.steps) and self.steps[self.step]["kind"] == "frequency")

    def _graph_x_range(self, kind):
        if kind == "psd":
            response = self.random_tab.response()
            if response is not None:
                return response.f_range
        xs = [c["x"] for c in self.curves if len(c["x"])]
        if xs:
            return float(min(np.nanmin(x) for x in xs)), float(max(np.nanmax(x) for x in xs))
        if self.steps:
            x = [self.steps[i]["x"] for i in self._series_steps()]
            return float(min(x)), float(max(x))
        return None

    def _draw_mode_lines(self, ax, kind, x_range):
        """Natural frequencies (eigenvalue summary) as vertical lines in a frequency graph."""
        modes = getattr(self.h5, "modes", None)
        if not modes or x_range is None or not self.modes_check.isChecked() or not self._frequency_axis(kind):
            return
        freq, numbers = modes["freq"], modes["mode"]
        inside = (freq >= x_range[0]) & (freq <= x_range[1])
        label_all = np.count_nonzero(inside) <= 40
        transform = ax.get_xaxis_transform()
        for f, m in zip(freq[inside], numbers[inside]):
            highlighted = self.highlight_mode is not None and int(m) == self.highlight_mode
            color = "tab:purple" if highlighted else "tab:green"
            ax.axvline(f, color=color, linewidth=1.6 if highlighted else 0.7, linestyle="-" if highlighted else ":",
                       alpha=0.9 if highlighted else 0.6, zorder=0)
            if label_all or highlighted:
                text = f" mode {m}: {f:.4g} Hz" if highlighted else f" {m}"
                ax.text(f, 0.98, text, transform=transform, fontsize=7 if highlighted else 6, color=color,
                        rotation=90, ha="right", va="top", clip_on=True)

    def _update_graph(self):
        if getattr(self, "graph_widget", None) is None or self._disposed:
            return
        name, location = self.results[self.idx]
        kind = self._graph_kind()
        current = None
        if self.selected is not None:
            node, cell = self.selected
            index = node if location == "point" else cell
            if index is not None:
                where = self._where_text(node, cell, location)
                if kind == "psd":
                    curve = self.random_tab.psd_curve(name, index)
                    if curve is not None:
                        current = dict(key=("psd", name, index, curve["id"]), label=f"{where}: {curve['label']}",
                                       steps=curve["steps"], x=curve["x"], y=curve["y"], kind="psd",
                                       unit=curve["unit"])
                else:
                    history = self._history(name, index)
                    if history is not None:
                        steps, x, y = history
                        current = dict(key=("history", name, index, self.cmode), label=f"{where}: {self._title(name)}",
                                       steps=steps, x=x, y=y, kind="history", unit=None)
        self.curves = [c for c in self.curves if c["kind"] == kind]
        if current is not None:
            if self.hold_check.isChecked():
                if current["key"] not in [c["key"] for c in self.curves]:
                    self.curves.append(current)
            else:
                self.curves = [current]
        elif not self.hold_check.isChecked():
            self.curves = []

        self.figure.clear()
        ax = self.axes = self.figure.add_subplot(111)
        self._twin = None
        for curve in self.curves:
            ax.plot(curve["x"], curve["y"], "-", marker=".", markersize=3, linewidth=1.2, label=curve["label"])
            if self.step in curve["steps"]:                     # marker = current step
                i = int(np.flatnonzero(curve["steps"] == self.step)[0])
                if np.isfinite(curve["y"][i]) and (kind != "psd" or curve["y"][i] > 0):
                    ax.plot(curve["x"][i], curve["y"][i], "o", color="red", markersize=6)
        x_range = self._graph_x_range(kind)
        if self.steps:
            x_step = self.steps[self.step]["x"]
            if x_range is None or x_range[0] <= x_step <= x_range[1]:
                ax.axvline(x_step, color="red", linewidth=0.8, linestyle="--")
        if kind == "psd":
            self._draw_input_psd(ax)
            ax.set_xscale("log")
            ax.set_yscale("log", nonpositive="mask")
            ax.set_xlabel("Frequency [Hz]")
            unit = self.curves[-1]["unit"] if self.curves else "g"
            ax.set_ylabel(f"PSD [{unit or '(model unit)'}$^2$/Hz]", fontsize=8)
            if x_range is not None and x_range[1] > x_range[0]:
                ax.set_xlim(*x_range)
        else:
            ax.set_xlabel(self.h5.step_axis(self.step))
            ax.set_ylabel(self._title(name), fontsize=8)
            if self.logy_check.isChecked() and any(np.nanmax(c["y"]) > 0 for c in self.curves if c["y"].size):
                ax.set_yscale("log", nonpositive="mask")
        self._draw_mode_lines(ax, kind, x_range)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        if self._twin is not None:
            more = self._twin.get_legend_handles_labels()
            handles, labels = handles + more[0], labels + more[1]
        if len(handles) > 1 or (kind == "psd" and handles):
            ax.legend(handles, labels, fontsize=7)
        if self.curves:
            hint = "response PSD, click = show that frequency" if kind == "psd" else "click into the graph = show that step"
            self.graph_label.setText(f"<b>{self.curves[-1]['label']}</b>  ({hint})")
        elif kind == "psd" and self.random_tab.response() is None:
            self.graph_label.setText(f"Random response is not available: {self.random_tab.error_text()}")
        elif self.selected is None:
            self.graph_label.setText("Select a node (double-click in 3D, key I or node ID) to see the course of "
                                     "the current result" + (" (response PSD)." if kind == "psd" else
                                                             " over the steps."))
        else:
            self.graph_label.setText(f"No course of {name} in the selected node / element.")
        try:
            self.figure.tight_layout()
        except Exception:  # noqa: BLE001  (e.g. too small canvas)
            pass
        self.canvas.draw_idle()

    def _draw_input_psd(self, ax):
        """Input PSD: in the same axes for accelerations in g (same unit), else on a secondary axis."""
        f, w = self.random_tab.input_curve()
        if f is None:
            return
        same_unit = all(c["unit"] == "g" for c in self.curves)
        target = ax if same_unit else ax.twinx()
        if target is not ax:
            self._twin = target
            target.set_yscale("log", nonpositive="mask")
            target.set_ylabel("Input PSD [g$^2$/Hz]", fontsize=8, color="gray")
            target.tick_params(labelsize=7, colors="gray")
        target.plot(f, w, "--", color="gray", linewidth=1.0, label=f"input PSD ({self.random_tab.input_text()})")

    def _on_graph_click(self, event):
        if event.inaxes is None or event.inaxes not in (self.axes, self._twin) or event.button != 1 \
                or event.xdata is None:
            return
        if str(getattr(self.graph_toolbar, "mode", "")):        # zoom / pan of the graph is active
            return
        candidates = self._series_steps()
        if self._graph_kind() == "psd":
            response = self.random_tab.response()
            candidates = list(response.steps) if response is not None else candidates
        if not candidates:
            return
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
        x_label = "Frequency [Hz]" if self.curves and self.curves[0]["kind"] == "psd" else self.h5.step_axis(self.step)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(sum([[x_label, c["label"]] for c in self.curves], []))
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
        self.surface = self._prepared["surface"]
        self.surface_points0 = self.surface.points.copy()
        self.surface_ids = np.asarray(self.surface.point_data["vtkOriginalPointIds"])
        self.surface_cells = np.asarray(self.surface.cell_data["_cell_index"])
        self.outline = self._prepared["outline"]                # model outline shown with sections
        self.outline_points0 = self.outline.points.copy()
        self.outline_ids = np.asarray(self.outline.point_data["vtkOriginalPointIds"])
        self.outline_actor = self.pl.add_mesh(self.outline, color="gray", opacity=0.5, line_width=1,
                                              name="outline", pickable=False, reset_camera=False, render=False)
        self.outline_actor.SetVisibility(False)

    def _load_step(self):
        """Deformation vectors of the current step (complex for frequency response)."""
        vectors = self.h5.deformation_vectors(self.step)
        if vectors is None:
            vectors = np.zeros((self.grid.n_points, 3))
        vectors = np.nan_to_num(np.asarray(vectors))
        self.U_complex = bool(np.iscomplexobj(vectors))
        self.Uc = vectors.astype(complex)
        self.u_max = float(np.sqrt((np.abs(self.Uc) ** 2).sum(axis=1)).max()) if len(self.Uc) else 0.0
        self.U = self._deformed(self.Uc, animated=False)       # shape at the deformation phase (probe, labels)

    def _phase_angle(self, animated=True):
        """Phase [rad] of the shown deformed shape: deformation phase of complex results + animation."""
        phase = np.radians(self.phase) if self.U_complex else 0.0
        return phase + (self.anim_phase if animated else 0.0)

    def _deformed(self, vectors, animated=True):
        """Real part of U * exp(i * phase) (real results: U * cos(animation phase))."""
        return (vectors * np.exp(1j * self._phase_angle(animated))).real

    def _update_mesh(self):
        """Mesh to display (outer surface / surface of the cut model / section plane only) with the values of
        the current result ("_value") and the (complex) deformation vectors."""
        name, location = self.results[self.idx]
        values = self._values()
        if not self.section:
            self.mesh = self.surface
            self.mesh_points0 = self.surface_points0
            self.mesh_Uc = self.Uc[self.surface_ids]
            self.mesh.point_data.pop("_value", None)
            self.mesh.cell_data.pop("_value", None)
            if location == "point":
                self.mesh.point_data["_value"] = values[self.surface_ids]
            else:
                self.mesh.cell_data["_value"] = values[self.surface_cells]
            return

        work = pv.UnstructuredGrid()
        work.ShallowCopy(self.grid)
        work.point_data["_Ure"] = self.Uc.real
        work.point_data["_Uim"] = self.Uc.imag
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
        if self.mesh.n_points:
            self.mesh_Uc = np.asarray(self.mesh.point_data["_Ure"]) + 1j * np.asarray(self.mesh.point_data["_Uim"])
        else:
            self.mesh_Uc = np.zeros((0, 3), dtype=complex)

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

    def _title(self, name):
        """Result name + complex part (+ sigma level of random RMS maps)."""
        title = name + (f" [{self.cmode}]" if self.h5.is_complex(name) else "")
        if self.random_tab is not None:
            title += self.random_tab.title_suffix(name)
        return title

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
            self._update_tree_ranges()
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
        self.U = self._deformed(self.Uc, animated=False)
        self._apply_deformation()
        self._update_info()
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
            title = self._title(name)
            suffix = self.random_tab.title_suffix(name) if self.random_tab is not None else ""
            if suffix:                                  # random RMS map: sigma level on the 2nd line
                title = title[:-len(suffix)] + "\n" + suffix.strip()
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
        rotation = np.exp(1j * self._phase_angle())
        if self.mesh.n_points > 0:
            self.mesh.points = self.mesh_points0 + self.scale * (self.mesh_Uc * rotation).real
        self.outline.points = self.outline_points0 + self.scale * (self.Uc[self.outline_ids] * rotation).real

    def set_animation(self, on=True):
        """Animation of the deformed shape (phase 0 - 360 deg, period by the spin box) on / off."""
        on = bool(on) and not self._disposed
        if on and self.u_max == 0:
            on = False
            self._status("Nothing to animate - no displacement in this step")
        if on and self.scale == 0:                              # no deformation shown yet -> 5 % of the model
            pct = self.deform_pct if self.deform_pct > 0 else 5.0
            self.deform_slider.blockSignals(True)
            self.deform_slider.setValue(int(round(pct * 10)))
            self.deform_slider.blockSignals(False)
            self._on_deform_slider(pct)
        self.animate_button.blockSignals(True)
        self.animate_button.setChecked(on)
        self.animate_button.blockSignals(False)
        if on:
            self._anim_t0 = time.perf_counter()
            self.anim_timer.start()
        else:
            self.anim_timer.stop()
            if self.anim_phase != 0.0 and not self._disposed:
                self.anim_phase = 0.0
                self._apply_deformation()
                self.pl.render()

    def _animation_frame(self):
        if self._disposed:
            self.anim_timer.stop()
            return
        period = max(self.period_spin.value(), 0.1)
        self.anim_phase = 2.0 * np.pi * (((time.perf_counter() - self._anim_t0) / period) % 1.0)
        self._apply_deformation()
        self.pl.render()

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
        positions = self.grid.points[nodes] + self.scale * self._deformed(self.Uc[nodes])
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
        title = self._title(name)
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
                         + (f", phase {self.phase:.0f} deg" if self.U_complex else ""))
        if self.selected is not None:
            node, cell = self.selected
            index = node if location == "point" else cell
            if index is not None:
                lines.append(f"Selected {self._where_text(node, cell, location)}: {_fmt(values[index])}")
        self.pl.add_text("\n".join(line for line in lines if line), position="upper_left", font_size=10,
                         color="black", name="info")

    # ================================================================================================= camera
    def camera_view(self):
        """Camera of the 3D view as a dict (saved in the settings)."""
        camera = self.pl.renderer.GetActiveCamera()
        return dict(position=list(camera.GetPosition()), focal_point=list(camera.GetFocalPoint()),
                    view_up=list(camera.GetViewUp()), view_angle=camera.GetViewAngle(),
                    parallel=bool(camera.GetParallelProjection()), parallel_scale=camera.GetParallelScale())

    def set_camera_view(self, view):
        """Restores a camera view saved by camera_view()."""
        camera = self.pl.renderer.GetActiveCamera()
        camera.SetPosition(*view["position"])
        camera.SetFocalPoint(*view["focal_point"])
        camera.SetViewUp(*view["view_up"])
        if "view_angle" in view:
            camera.SetViewAngle(view["view_angle"])
        camera.SetParallelProjection(bool(view.get("parallel", False)))
        if "parallel_scale" in view:
            camera.SetParallelScale(view["parallel_scale"])
        self.pl.renderer.ResetCameraClippingRange()
        self.pl.render()

    # ================================================================================================= run
    def show(self):
        """Opens the window and runs the Qt event loop (returns after the window is closed)."""
        self.window.show()
        self.pl.render()
        run_app()

    def screenshot(self, file_path):
        """Saves the 3D view to a PNG file."""
        self.pl.render()
        self.pl.screenshot(str(file_path))

    def dispose(self):
        """Stops the animation and background tasks, closes the 3D view and the H5 file (another file is opened
        in the window or the window is closed)."""
        if self._disposed:
            return
        self._disposed = True
        self.anim_timer.stop()
        self.tasks.stop(wait=2.0)
        try:
            self.pl.close()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        if hasattr(self.h5, "close"):
            safe_call(self.h5.close)
