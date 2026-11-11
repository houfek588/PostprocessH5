"""
3D viewer of a Nastran input model (BdfModel) - Qt window based on the H5 result viewer
(visualization/h5_viewer.py: table of fields, sections, probe, screenshots) extended by:
    * model entities in 3D (SPC / SPCD nodes, rigid elements, CONM2 masses, gravity, forces) with checkboxes
    * tab "Report" with the model report (summary, mass, case control, loads, mesh checks)
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]                      # project root -> visualization package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from visualization.h5_viewer import H5ResultViewer  # noqa: E402
from qtpy import QtGui, QtWidgets  # noqa: E402  (available - checked by h5_viewer)


class BdfViewer(H5ResultViewer):
    def __init__(self, model, cmap="jet", show_edges=True, window_size=(1650, 950)):
        """
        Args:
            model (BdfModel): analysed Nastran input model
        """
        self.model = model
        self._entity_actors = {}                                # key -> [actors]
        self._entity_labels = {}                                # key -> label actor
        self._entity_visible = {key: True for key in model.entities}
        n_labels = sum(len(e.get("texts") or []) for e in model.entities.values())
        self.show_labels = n_labels <= 10                       # many labels -> off at start
        super().__init__(model, cmap=cmap, show_edges=show_edges, window_size=window_size)
        self.window.setWindowTitle(f"Nastran model - {model.file_path.name}")

    # ================================================================================================= panel
    def _build_panel(self):
        panel = super()._build_panel()                          # fields, display, section, selection
        layout = panel.layout()

        # ---- model entities (checkboxes) - inserted above the probe
        if self.model.entities:
            group = QtWidgets.QGroupBox("Model entities")
            box = QtWidgets.QVBoxLayout(group)
            for key, entity in self.model.entities.items():
                check = QtWidgets.QCheckBox(f"{entity['label']}  [{entity['count']}]")
                check.setChecked(True)
                color = pv.Color(entity["color"]).hex_rgb
                check.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold }}")
                check.toggled.connect(lambda on, k=key: self.set_entity_visible(k, on))
                box.addWidget(check)
            if any(e.get("texts") for e in self.model.entities.values()):
                labels = QtWidgets.QCheckBox("Labels (IDs, values)")
                labels.setChecked(self.show_labels)
                labels.toggled.connect(self.set_labels_visible)
                box.addWidget(labels)
            layout.insertWidget(layout.indexOf(self.selection_group), group)

        # ---- report tab
        report = QtWidgets.QWidget()
        report_layout = QtWidgets.QVBoxLayout(report)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
                             if hasattr(QtWidgets.QPlainTextEdit, "LineWrapMode")
                             else QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont
                                              if hasattr(QtGui.QFontDatabase, "SystemFont")
                                              else QtGui.QFontDatabase.FixedFont)
        text.setFont(font)
        text.setPlainText(self.model.report)
        save = QtWidgets.QPushButton("Save report...")
        save.clicked.connect(lambda checked=False: self._on_save_report())
        report_layout.addWidget(text)
        report_layout.addWidget(save)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(panel, "Model")
        tabs.addTab(report, "Report")
        return tabs

    def _on_save_report(self):
        default = self.model.file_path.with_name(self.model.file_path.stem + "_report.txt")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Save report", str(default),
                                                        "Text file (*.txt)")
        if path:
            self.model.save_report(path)

    # ================================================================================================= 3D entities
    def _build_scene(self):
        super()._build_scene()
        for key, entity in self.model.entities.items():
            self._entity_actors[key] = self._add_entity(key, entity)
        self._update_entity_visibility()

    def _add_entity(self, key, entity):
        actors = []
        color = entity["color"]
        size = self.model_size
        common = dict(pickable=False, reset_camera=False, render=False)
        if entity["kind"] == "points":
            actors.append(self.pl.add_points(pv.PolyData(np.asarray(entity["points"], dtype=float)), color=color,
                                             point_size=12, render_points_as_spheres=True,
                                             name=f"entity_{key}", **common))
        elif entity["kind"] == "lines":
            segments = np.asarray(entity["lines"], dtype=float)         # (n, 2, 3)
            points = segments.reshape(-1, 3)
            cells = np.column_stack([np.full(len(segments), 2), np.arange(0, 2 * len(segments), 2),
                                     np.arange(1, 2 * len(segments), 2)]).ravel()
            actors.append(self.pl.add_mesh(pv.PolyData(points, lines=cells), color=color, line_width=2,
                                           name=f"entity_{key}_lines", **common))
            actors.append(self.pl.add_points(pv.PolyData(np.asarray(entity["points"], dtype=float)), color=color,
                                             point_size=10, render_points_as_spheres=True,
                                             name=f"entity_{key}", **common))
        elif entity["kind"] == "arrows":
            arrows = None
            for start, vector in zip(np.asarray(entity["points"], float), np.asarray(entity["vectors"], float)):
                length = float(np.linalg.norm(vector))
                if length == 0:
                    continue
                arrow = pv.Arrow(start=start, direction=vector, scale=length, tip_length=0.25,
                                 tip_radius=0.08, shaft_radius=0.03)
                arrows = arrow if arrows is None else arrows.merge(arrow)
            if arrows is not None:
                actors.append(self.pl.add_mesh(arrows, color=color, name=f"entity_{key}", **common))
        texts = entity.get("texts")
        if texts:
            positions = np.asarray(entity["points"], dtype=float) + np.array([0.0, 0.0, 0.01 * size])
            self._entity_labels[key] = self.pl.add_point_labels(
                positions, texts, name=f"entity_{key}_labels", font_size=10, text_color="black",
                shape_color="white", shape_opacity=0.7, show_points=False, always_visible=True, render=False)
        return [a for a in actors if a is not None]

    def _update_entity_visibility(self):
        for key, actors in self._entity_actors.items():
            for actor in actors:
                actor.SetVisibility(self._entity_visible[key])
        for key, actor in self._entity_labels.items():
            if actor is not None:
                actor.SetVisibility(self._entity_visible[key] and self.show_labels)
        self.pl.render()

    def set_entity_visible(self, key, visible=True):
        self._entity_visible[key] = bool(visible)
        self._update_entity_visibility()

    def set_labels_visible(self, visible=True):
        self.show_labels = bool(visible)
        self._update_entity_visibility()
