"""
Tabs of the H5 viewer for dynamic results (visualization/h5_viewer.py):

RandomTab - random vibration from a frequency response with enforced motion (SOL 111 / 108): input acceleration PSD
            (presets, editable table, CSV), reference input (acceleration of an enforced node or constant
            amplitude), g in model units; response PSD of the selected node in the graph; RMS (1 / 2 / 3 sigma) maps
            of displacement, acceleration, ... and stress (von Mises by Segalman) computed in the background.
            Computation: file_processing/random_response.py
ModesTab  - natural frequencies of the eigenvalue summary (/NASTRAN/RESULT/SUMMARY/EIGENVALUE): table of modes,
            click = mode shape (EIGENVECTOR results) or the nearest frequency of the frequency response; the modes
            are shown as vertical lines in the frequency graph.
"""
import numpy as np

from file_processing.random_response import (G_VALUES, PRESETS, RandomResponse, check_psd_table, guess_g, psd_rms,
                                             read_psd_csv, write_psd_csv)
from visualization.qt_base import USER_ROLE, QtWidgets, _class_enum, safe_call

NO_EDIT = _class_enum(QtWidgets.QAbstractItemView, "EditTrigger", "NoEditTriggers")
SELECT_ROWS = _class_enum(QtWidgets.QAbstractItemView, "SelectionBehavior", "SelectRows")
SINGLE_SELECTION = _class_enum(QtWidgets.QAbstractItemView, "SelectionMode", "SingleSelection")


def _number(text):
    """Number from a text field (decimal point or comma)."""
    return float(str(text).strip().replace(",", "."))


def _scroll(page):
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(page)
    return area


# =====================================================================================================================
#   random vibration
# =====================================================================================================================
class RandomTab:
    LEVELS = ("1 sigma (= RMS)", "2 sigma", "3 sigma")

    @staticmethod
    def available(h5):
        return hasattr(h5, "nodal_history") and bool(RandomResponse.frequency_steps(h5))

    def __init__(self, viewer):
        self.viewer = viewer
        self.h5 = viewer.h5
        self.series_steps = RandomResponse.frequency_steps(self.h5)     # first step of each FRF subcase
        self.maps = {}                  # result name -> map (values = 1 sigma, source for the PSD)
        self.level = 1                  # shown sigma level of the maps
        self.g_reason = ""
        self._response = None
        self._error = None
        self._version = 0               # increased with every change of the settings
        self._maps_version = None
        self._psd_cache = {}
        try:
            self.spcd = self.h5.enforced_motion()
        except Exception:  # noqa: BLE001
            self.spcd = []
        self.widget = _scroll(self._build())
        self._fill_table(PRESETS[next(iter(PRESETS))])
        self._auto_reference()
        self._auto_g()
        self._settings_changed(notify=False)

    # ---------------------------------------------------------------------------------------------- widgets
    def _build(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---- input PSD
        group = QtWidgets.QGroupBox("Input acceleration PSD [g²/Hz]")
        box = QtWidgets.QVBoxLayout(group)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS) + ["Custom"])
        self.preset_combo.currentIndexChanged.connect(lambda i: safe_call(self._on_preset, i))
        box.addWidget(self.preset_combo)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Frequency [Hz]", "PSD [g²/Hz]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(130)
        self.table.itemChanged.connect(lambda item: safe_call(self._on_table_edit))
        box.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        for text, callback in (("Add row", self._add_row), ("Remove row", self._remove_row),
                               ("Load...", self._on_load), ("Save...", self._on_save)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, c=callback: safe_call(c))
            row.addWidget(button)
        box.addLayout(row)
        self.input_label = QtWidgets.QLabel()
        self.input_label.setWordWrap(True)
        box.addWidget(self.input_label)
        layout.addWidget(group)

        # ---- excitation of the frequency response
        group = QtWidgets.QGroupBox("Input of the frequency response (reference)")
        grid = QtWidgets.QGridLayout(group)
        r = 0
        self.subcase_combo = None
        if len(self.series_steps) > 1:
            self.subcase_combo = QtWidgets.QComboBox()
            for step in self.series_steps:
                self.subcase_combo.addItem(f"Subcase {self.h5.steps[step]['subcase']}")
            self.subcase_combo.currentIndexChanged.connect(lambda i: safe_call(self._settings_changed))
            grid.addWidget(QtWidgets.QLabel("Subcase:"), r, 0)
            grid.addWidget(self.subcase_combo, r, 1, 1, 3)
            r += 1
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems(["Acceleration of a node (enforced motion)", "Constant amplitude"])
        self.ref_combo.setToolTip("Amplitude of the harmonic input acceleration of the frequency response "
                                  "(the transfer function = response / this amplitude)")
        self.ref_combo.currentIndexChanged.connect(lambda i: safe_call(self._on_reference_kind))
        grid.addWidget(QtWidgets.QLabel("Input:"), r, 0)
        grid.addWidget(self.ref_combo, r, 1, 1, 3)
        r += 1
        self.ref_node_edit = QtWidgets.QLineEdit()
        self.ref_node_edit.setToolTip("Node with the enforced motion (SPCD), its acceleration is the input")
        self.ref_node_edit.editingFinished.connect(lambda: safe_call(self._settings_changed))
        self.ref_dof_combo = QtWidgets.QComboBox()
        self.ref_dof_combo.addItems(["X", "Y", "Z"])
        self.ref_dof_combo.currentIndexChanged.connect(lambda i: safe_call(self._settings_changed))
        self.ref_node_label = QtWidgets.QLabel("Node ID:")
        self.ref_dof_label = QtWidgets.QLabel("direction:")
        grid.addWidget(self.ref_node_label, r, 0)
        grid.addWidget(self.ref_node_edit, r, 1)
        grid.addWidget(self.ref_dof_label, r, 2)
        grid.addWidget(self.ref_dof_combo, r, 3)
        r += 1
        self.ref_value_label = QtWidgets.QLabel("Amplitude:")
        self.ref_value_edit = QtWidgets.QLineEdit("1.0")
        self.ref_value_edit.setToolTip("Amplitude of the input acceleration in model units")
        self.ref_value_edit.editingFinished.connect(lambda: safe_call(self._settings_changed))
        grid.addWidget(self.ref_value_label, r, 0)
        grid.addWidget(self.ref_value_edit, r, 1, 1, 3)
        r += 1
        self.g_edit = QtWidgets.QLineEdit()
        self.g_edit.setToolTip("Gravity acceleration in model units (input PSD in g²/Hz, accelerations in g)")
        self.g_edit.editingFinished.connect(lambda: safe_call(self._on_g_edit))
        self.g_combo = QtWidgets.QComboBox()
        self.g_combo.addItem("units ...")
        for label, value in G_VALUES:
            self.g_combo.addItem(f"{value:g}  {label}")
        self.g_combo.currentIndexChanged.connect(lambda i: safe_call(self._on_g_combo, i))
        grid.addWidget(QtWidgets.QLabel("g ="), r, 0)
        grid.addWidget(self.g_edit, r, 1)
        grid.addWidget(self.g_combo, r, 2, 1, 2)
        r += 1
        self.ref_label = QtWidgets.QLabel()
        self.ref_label.setWordWrap(True)
        grid.addWidget(self.ref_label, r, 0, 1, 4)
        layout.addWidget(group)

        # ---- RMS maps
        group = QtWidgets.QGroupBox("RMS maps of the whole model (3D)")
        box = QtWidgets.QVBoxLayout(group)
        self.source_checks = []
        step = self.series_steps[0]
        domain = self.h5.steps[step]["domain"]
        for source, label in self._map_sources(domain):
            check = QtWidgets.QCheckBox(label)
            base = self.h5._tables[source[1]]["base"] if source[0] == "nodal" else source[1]
            check.setChecked(base in ("DISPLACEMENT", "ACCELERATION", "VELOCITY", "STRESS"))
            box.addWidget(check)
            self.source_checks.append((source, check))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show as:"))
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.addItems(self.LEVELS)
        self.level_combo.setToolTip("Gaussian response with zero mean: 1 sigma = RMS (68.3 %), 2 sigma (95.4 %), "
                                    "3 sigma = 3 x RMS (99.73 % of the time below)")
        self.level_combo.currentIndexChanged.connect(lambda i: safe_call(self.set_level, i + 1))
        row.addWidget(self.level_combo)
        self.compute_button = QtWidgets.QPushButton("Compute RMS maps")
        self.compute_button.clicked.connect(lambda checked=False: safe_call(self.compute))
        row.addWidget(self.compute_button)
        box.addLayout(row)
        self.maps_label = QtWidgets.QLabel("Maps are computed in the background, then they are in the table of "
                                           "results (tab Results, group 'Random ...').")
        self.maps_label.setWordWrap(True)
        box.addWidget(self.maps_label)
        layout.addWidget(group)

        note = QtWidgets.QLabel(
            "Response PSD = |H|² · g² · PSD<sub>in</sub>, H = response / input amplitude "
            "(accelerations in g). RMS = √∫PSD df (trapezoidal rule over the frequencies of the frequency "
            "response, only where the input PSD is defined). Magnitude = √(Σ RMS²), von Mises RMS "
            "by Segalman. One input, linear, Gaussian, zero mean.<br>"
            "<b>Graph:</b> select a node - RMS maps show its response PSD, frequency response results too with "
            "'Response PSD' checked.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _map_sources(self, domain):
        out = []
        for key, info in self.h5._tables.items():
            if info["kind"] == "nodal" and info["complex"] and domain in info["blocks"]:
                label = self.h5._nodal_label(info["base"]) + (" [g]" if info["base"] == "ACCELERATION" else "")
                out.append((("nodal", key), label))
        for base in ("STRESS", "STRAIN"):
            if any(i["kind"] == "element" and i["base"] == base and i["complex"] and domain in i["blocks"]
                   for i in self.h5._tables.values()):
                extra = "von Mises + components" if base == "STRESS" else "components"
                out.append((("element", base), f"{base.capitalize()} ({extra}, element + nodal avg.)"))
        return out

    # ---------------------------------------------------------------------------------------------- PSD table
    def _fill_table(self, table):
        self.table.blockSignals(True)
        self.table.setRowCount(len(table))
        for r, (f, w) in enumerate(table):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{f:g}"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{w:g}"))
        self.table.blockSignals(False)

    def psd_table(self):
        """PSD table from the widget -> array (n, 2) (ValueError for invalid input)."""
        rows = []
        for r in range(self.table.rowCount()):
            texts = [self.table.item(r, c).text() if self.table.item(r, c) is not None else "" for c in (0, 1)]
            if not any(t.strip() for t in texts):
                continue                                    # empty row
            try:
                rows.append((_number(texts[0]), _number(texts[1])))
            except ValueError:
                raise ValueError(f"PSD table, row {r + 1}: invalid number") from None
        return check_psd_table(rows)

    def set_psd_table(self, table, preset=None):
        """Sets the input PSD (preset = its name in the list or None = custom)."""
        self._fill_table(check_psd_table(table))
        self.preset_combo.blockSignals(True)
        names = list(PRESETS)
        self.preset_combo.setCurrentIndex(names.index(preset) if preset in names else len(names))
        self.preset_combo.blockSignals(False)
        self._settings_changed()

    def _on_preset(self, index):
        names = list(PRESETS)
        if index < len(names):
            self._fill_table(PRESETS[names[index]])
            self._settings_changed()

    def _on_table_edit(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(len(PRESETS))     # Custom
        self.preset_combo.blockSignals(False)
        self._settings_changed()

    def _add_row(self):
        self.table.insertRow(self.table.rowCount())

    def _remove_row(self):
        row = self.table.currentRow()
        self.table.removeRow(row if row >= 0 else self.table.rowCount() - 1)
        self._on_table_edit()

    def _on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.viewer.window, "Load input PSD",
                                                        str(self.h5.file_path.parent),
                                                        "PSD table (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            self.set_psd_table(read_psd_csv(path))
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self.viewer.window, "Load input PSD", str(error))

    def _on_save(self):
        try:
            table = self.psd_table()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self.viewer.window, "Save input PSD", str(error))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.viewer.window, "Save input PSD",
                                                        str(self.h5.file_path.with_name("input_psd.csv")),
                                                        "CSV file (*.csv)")
        if path:
            write_psd_csv(path, table)

    # ---------------------------------------------------------------------------------------------- reference, g
    def _auto_reference(self):
        """Reference = acceleration of the first node with enforced motion (SPCD) in its direction."""
        has_acceleration = RandomResponse.acceleration_table(self.h5) is not None
        node, dof = None, 1
        for g, components, _ in self.spcd:
            digits = [int(c) for c in components if c in "123"]
            if digits and self.h5.node_index([g])[0] >= 0:
                node, dof = g, digits[0]
                break
        self.ref_node_edit.setText("" if node is None else str(node))
        self.ref_dof_combo.blockSignals(True)
        self.ref_dof_combo.setCurrentIndex(dof - 1)
        self.ref_dof_combo.blockSignals(False)
        self.ref_combo.blockSignals(True)
        self.ref_combo.setCurrentIndex(0 if has_acceleration and node is not None else 1)
        self.ref_combo.blockSignals(False)
        self._update_reference_widgets()

    def _auto_g(self):
        try:
            amplitude = RandomResponse.reference_amplitude(self.h5, self.reference(), self._domains())
        except (ValueError, KeyError, IndexError):
            amplitude = None
        g, self.g_reason = guess_g(amplitude, self.h5)
        self.g_edit.setText(f"{g:.6g}")

    def _update_reference_widgets(self):
        node = self.ref_combo.currentIndex() == 0
        for widget in (self.ref_node_label, self.ref_node_edit, self.ref_dof_label, self.ref_dof_combo):
            widget.setVisible(node)
        for widget in (self.ref_value_label, self.ref_value_edit):
            widget.setVisible(not node)

    def _on_reference_kind(self):
        self._update_reference_widgets()
        self._settings_changed()

    def _on_g_edit(self):
        self.g_reason = "entered"
        self._settings_changed()

    def _on_g_combo(self, index):
        if index > 0:
            label, value = G_VALUES[index - 1]
            self.g_edit.setText(f"{value:g}")
            self.g_reason = label
            self.g_combo.blockSignals(True)
            self.g_combo.setCurrentIndex(0)
            self.g_combo.blockSignals(False)
            self._settings_changed()

    def reference(self):
        """("node", node ID, component) or ("constant", amplitude) from the widgets (ValueError if invalid)."""
        if self.ref_combo.currentIndex() == 0:
            try:
                node = int(self.ref_node_edit.text().strip())
            except ValueError:
                raise ValueError("Enter the ID of the node with the enforced motion") from None
            return ("node", node, self.ref_dof_combo.currentIndex() + 1)
        try:
            return ("constant", _number(self.ref_value_edit.text()))
        except ValueError:
            raise ValueError("Invalid amplitude of the input") from None

    def g(self):
        try:
            value = _number(self.g_edit.text())
        except ValueError:
            raise ValueError("Invalid value of g") from None
        if value <= 0:
            raise ValueError("g must be > 0")
        return value

    def _series_step(self):
        return self.series_steps[self.subcase_combo.currentIndex() if self.subcase_combo is not None else 0]

    def _domains(self):
        return [self.h5.steps[i]["domain"] for i in self.h5.series_steps(self._series_step())]

    # ---------------------------------------------------------------------------------------------- response
    def response(self):
        """RandomResponse for the current settings (None if they are invalid -> error_text())."""
        if self._response is None and self._error is None:
            try:
                self._response = RandomResponse(self.h5, self.psd_table(), self.g(), self.reference(),
                                                self._series_step())
            except (ValueError, KeyError, IndexError) as error:
                self._error = str(error)
        return self._response

    def error_text(self):
        self.response()
        return self._error or ""

    def _settings_changed(self, notify=True):
        self._version += 1
        self._response = None
        self._error = None
        self._psd_cache.clear()
        response = self.response()
        self._update_labels(response)
        if notify:
            self.viewer._update_graph()

    def _update_labels(self, response):
        try:
            table = self.psd_table()
            text = (f"Input: <b>{psd_rms(table):.3g} Grms</b> ({table[0, 0]:g} - {table[-1, 0]:g} Hz)")
        except ValueError as error:
            table = None
            text = f"<span style='color:red'>{error}</span>"
        if response is not None:
            f0, f1 = response.f_range
            text += (f"<br>Frequency response {response.freq.min():.4g} - {response.freq.max():.4g} Hz, "
                     f"{len(response.freq)} frequencies -> RMS over {f0:.4g} - {f1:.4g} Hz: input "
                     f"{response.table_rms_range:.3g} Grms (trapezoidal at the FRF frequencies "
                     f"{response.input_rms:.3g} Grms)")
            if table is not None and (table[0, 0] < response.freq.min() * 0.999
                                      or table[-1, 0] > response.freq.max() * 1.001):
                text += ("<br><span style='color:#c05000'>The input PSD is wider than the frequency response - "
                         "the part outside of it is not included.</span>")
        self.input_label.setText(text)

        if response is None:
            self.ref_label.setText(f"<span style='color:red'>{self._error}</span>")
        else:
            amplitude = np.abs(response.ref[response.psd_in > 0])
            low, high = float(amplitude.min()), float(amplitude.max())
            value = f"{low:.5g}" if np.isclose(low, high, rtol=1e-3) else f"{low:.5g} - {high:.5g}"
            in_g = f"{low / response.g:.4g}" if np.isclose(low, high, rtol=1e-3) else \
                f"{low / response.g:.4g} - {high / response.g:.4g}"
            text = f"Input amplitude {value} = <b>{in_g} g</b>  (g = {response.g:g}: {self.g_reason})"
            df = np.diff(response.freq)
            modes = getattr(self.h5, "modes", {}).get("freq")
            if df.size and modes is not None:
                inside = modes[(modes >= response.f_range[0]) & (modes <= response.f_range[1])]
                if inside.size and np.median(df) > 0.01 * inside.min():
                    text += (f"<br><span style='color:#c05000'>Frequency step {np.median(df):.3g} Hz is coarse "
                             f"for the modes from {inside.min():.4g} Hz - resonance peaks may be missed and the "
                             f"RMS is inaccurate (FREQ3 / FREQ4 around the modes recommended).</span>")
            self.ref_label.setText(text)
        if self.maps and self._maps_version != self._version:
            self.maps_label.setText("<span style='color:#c05000'>Settings were changed - press 'Compute RMS "
                                    "maps' to update the maps.</span> The response PSD in the graph uses the "
                                    "new settings.")

    # ---------------------------------------------------------------------------------------------- viewer API
    def is_random_result(self, name):
        return name in self.maps

    def supports(self, name):
        """Can the result be shown as a response PSD (complex result of the frequency response)?"""
        response = self.response()
        return response is not None and response.source_of_quantity(name) is not None

    def title_suffix(self, name):
        if name not in self.maps:
            return ""
        return " (1 sigma = RMS)" if self.level == 1 else f" ({self.level} sigma = {self.level} x RMS)"

    def input_curve(self):
        response = self.response()
        return response.input_curve() if response is not None else (None, None)

    def input_text(self):
        response = self.response()
        return f"{response.input_rms:.3g} Grms" if response is not None else ""

    def psd_curve(self, name, index):
        """Response PSD of a result in node / element `index` -> dict for the graph or None."""
        response = self.response()
        if response is None:
            return None
        if name in self.maps:
            source = self.maps[name]["source"]
            label = name[4:] if name.startswith("RMS ") else name
        else:
            source = response.source_of_quantity(name)
            label = name
        if source is None:
            return None
        key = (self._version, source, index)
        if key not in self._psd_cache:
            self._psd_cache[key] = response.psd(source, index)
        y = self._psd_cache[key]
        rms = response.rms(y)
        unit = response.unit(source)
        text = f"{label}: RMS {rms:.4g}" + (" g" if unit == "g" else "")
        if self.level > 1:
            text += f", {self.level} sigma {self.level * rms:.4g}"
        return dict(x=response.freq, y=y, steps=response.steps, unit=unit, label=text, id=self._version, rms=rms)

    # ---------------------------------------------------------------------------------------------- RMS maps
    def compute(self):
        """Computes the RMS maps of the checked results in the background."""
        response = self.response()
        if response is None:
            QtWidgets.QMessageBox.warning(self.viewer.window, "Random response", self._error or "Invalid settings")
            return
        sources = [source for source, check in self.source_checks if check.isChecked()]
        if not sources:
            self.maps_label.setText("<span style='color:red'>Check at least one result.</span>")
            return
        self.compute_button.setEnabled(False)
        self.maps_label.setText("Computing ... (progress in the status bar)")
        version = self._version
        self.viewer.tasks.start("random", lambda task: response.compute_maps(sources, task.progress),
                                done=lambda maps: self._on_maps(maps, version, response),
                                error=self._on_maps_error)

    def _on_maps(self, maps, version, response):
        self.compute_button.setEnabled(True)
        current = self.viewer.results[self.viewer.idx][0]
        for m in maps:
            self.maps[m["name"]] = m
        self._maps_version = version
        self._apply_level()
        select = maps[0]["name"] if maps and current not in self.maps else None
        self.viewer.results_changed(select=select)
        f0, f1 = response.f_range
        self.maps_label.setText(f"{len(maps)} RMS maps ({f0:.4g} - {f1:.4g} Hz, input {response.input_rms:.3g} "
                                f"Grms) - tab Results, groups 'Random ...'. Shown as {self.LEVELS[self.level - 1]}.")
        self.viewer._status(f"Random response: {len(maps)} RMS maps computed")

    def _on_maps_error(self, error):
        self.compute_button.setEnabled(True)
        self.maps_label.setText(f"<span style='color:red'>Computation failed: {error}</span>")

    def _apply_level(self):
        k = self.level
        for name, m in self.maps.items():
            self.h5.add_static_result(name, m["location"], f"Random {k}σ - {m['group']}", m["item"],
                                      m["values"] * k)

    def set_level(self, level):
        """Shown level of the RMS maps: 1, 2 or 3 sigma."""
        self.level = int(level)
        self.level_combo.blockSignals(True)
        self.level_combo.setCurrentIndex(self.level - 1)
        self.level_combo.blockSignals(False)
        if self.maps:
            self._apply_level()
            self.viewer.results_changed()
        else:
            self.viewer._update_graph()


# =====================================================================================================================
#   modes
# =====================================================================================================================
class ModesTab:
    @staticmethod
    def available(h5):
        modes = getattr(h5, "modes", None)
        return bool(modes) and len(modes.get("freq", [])) > 0

    def __init__(self, viewer):
        self.viewer = viewer
        self.modes = viewer.h5.modes
        steps = viewer.steps
        self.mode_steps = {int(s["mode"]): i for i, s in enumerate(steps) if s["kind"] == "mode"}
        self.freq_steps = [i for i, s in enumerate(steps) if s["kind"] == "frequency"]
        self.f_range = None
        if self.freq_steps:
            x = [steps[i]["x"] for i in self.freq_steps]
            self.f_range = (min(x), max(x))
        self.widget = self._build()

    def _build(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        freq = self.modes["freq"]
        text = f"<b>{len(freq)} modes</b> (eigenvalue summary), {freq.min():.4g} - {freq.max():.4g} Hz"
        if self.f_range is not None:
            inside = np.count_nonzero((freq >= self.f_range[0]) & (freq <= self.f_range[1]))
            text += (f"; {inside} in the range of the frequency response {self.f_range[0]:.4g} - "
                     f"{self.f_range[1]:.4g} Hz (green lines in the graph)")
        if self.mode_steps:
            text += ("<br>Mode shapes are in the file: click a mode = show it, <b>Animate</b> (tab Results) = "
                     "animation of the mode shape.")
        else:
            text += ("<br>Mode shapes are not in the file (no EIGENVECTOR output of the modal subcase): click a "
                     "mode = nearest frequency of the frequency response, <b>Animate</b> shows the deflection "
                     "shape there.")
        info = QtWidgets.QLabel(text)
        info.setWordWrap(True)
        layout.addWidget(info)
        self.range_check = QtWidgets.QCheckBox("Only modes in the frequency range of the results")
        self.range_check.setVisible(self.f_range is not None)
        self.range_check.setChecked(self.f_range is not None and len(freq) > 50)
        self.range_check.toggled.connect(lambda on: safe_call(self._fill))
        layout.addWidget(self.range_check)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Mode", "Frequency [Hz]", "Gen. mass", "Gen. stiffness"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(NO_EDIT)
        self.table.setSelectionBehavior(SELECT_ROWS)
        self.table.setSelectionMode(SINGLE_SELECTION)
        self.table.itemSelectionChanged.connect(lambda: safe_call(self._on_select))
        layout.addWidget(self.table, stretch=1)
        row = QtWidgets.QHBoxLayout()
        animate = QtWidgets.QPushButton("Animate selected")
        animate.setToolTip("Show the selected mode and start the animation")
        animate.clicked.connect(lambda checked=False: safe_call(self._on_animate))
        clear = QtWidgets.QPushButton("Clear highlight")
        clear.clicked.connect(lambda checked=False: safe_call(self.clear))
        row.addWidget(animate)
        row.addWidget(clear)
        layout.addLayout(row)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._fill()
        return page

    def _fill(self):
        freq = self.modes["freq"]
        rows = np.arange(len(freq))
        if self.range_check.isChecked() and self.f_range is not None:
            rows = rows[(freq >= self.f_range[0]) & (freq <= self.f_range[1])]
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, k in enumerate(rows):
            values = [f"{self.modes['mode'][k]}", f"{freq[k]:.5g}"]
            values += [f"{self.modes[key][k]:.4g}" if key in self.modes else "" for key in ("mass", "stiff")]
            for c, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(USER_ROLE, int(k))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def select_mode(self, mode):
        """Highlights the mode in the graph and shows its shape (or the nearest frequency of the FRF)."""
        k = int(np.flatnonzero(self.modes["mode"] == mode)[0])
        freq = float(self.modes["freq"][k])
        viewer = self.viewer
        viewer.highlight_mode = int(mode)
        if int(mode) in self.mode_steps:
            viewer.set_step(self.mode_steps[int(mode)])
            where = "mode shape"
            if not np.isfinite(viewer._values()).any():        # e.g. Displacement -> Eigenvector magnitude
                for i, (name, _) in enumerate(viewer.results):
                    if viewer.h5.result_meta.get(name, ("", ""))[0] == "Eigenvector":
                        viewer.set_result(i)
                        break
        elif self.freq_steps:
            current = viewer.steps[viewer.step]
            candidates = viewer._series_steps() if current["kind"] == "frequency" else \
                viewer._series_steps(self.freq_steps[0])
            x = np.array([viewer.steps[i]["x"] for i in candidates])
            viewer.set_step(candidates[int(np.argmin(np.abs(x - freq)))])
            where = "nearest frequency of the frequency response"
        else:
            where = "no frequency results"
        viewer._update_graph()
        self.status_label.setText(f"Mode {mode}: {freq:.5g} Hz -> {where}: {viewer.steps[viewer.step]['label']}"
                                  if viewer.steps else f"Mode {mode}: {freq:.5g} Hz")

    def _on_select(self):
        items = self.table.selectedItems()
        if items:
            k = int(items[0].data(USER_ROLE))
            self.select_mode(int(self.modes["mode"][k]))

    def _on_animate(self):
        if self.viewer.highlight_mode is None:
            self.status_label.setText("Select a mode in the table first.")
            return
        self.viewer.set_animation(True)

    def clear(self):
        self.viewer.highlight_mode = None
        self.table.clearSelection()
        self.status_label.setText("")
        self.viewer._update_graph()
