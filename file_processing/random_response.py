"""
Random vibration response computed from a frequency response analysis (SOL 111 / 108) with enforced motion.

Method
    The frequency response X(f) to a harmonic base excitation with the amplitude a_ref(f) (enforced acceleration by
    SPCD + RLOAD1 TYPE=ACCE, e.g. 1 g or 1.0) is scaled to the input acceleration PSD W_in(f) [g^2/Hz]:

        W_out(f) = |X(f) / a_ref(f)|^2 * g^2 * W_in(f)        [response units^2/Hz]
        accelerations are given in g:   W_out(f) = |A(f) / a_ref(f)|^2 * W_in(f)        [g^2/Hz]

        RMS = sqrt( integral of W_out(f) df )   - trapezoidal rule over the frequencies of the frequency response
                                                  (only where the input PSD is defined)

    vector magnitude:   RMS = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)
    von Mises stress:   PSD of the equivalent von Mises stress from the complex stress components (Segalman et al.):
                        |sx|^2 + |sy|^2 + |sz|^2 - Re(sx sy*) - Re(sy sz*) - Re(sz sx*) + 3 (|txy|^2 + |tyz|^2 + |tzx|^2)

    Assumptions: one excitation (all enforced nodes move together in one direction), linear system, stationary
    Gaussian response with zero mean -> 1 sigma = RMS, 3 sigma = 3 x RMS (99.73 % of the time).

Example:
    h5 = ParseH5File("random_ram_edit.h5")
    rr = RandomResponse(h5, PRESETS["GEVS qualification (14.1 Grms)"], g=386.4, reference=("node", 790, 2))
    f, psd = rr.freq, rr.psd(("nodal", "ACCELERATION_CPLX", [1]), h5.node_index([1234])[0])
    maps = rr.compute_maps([("nodal", "ACCELERATION_CPLX"), ("element", "STRESS")])
"""
import csv
import re
from pathlib import Path

import numpy as np

# input acceleration PSD tables [(frequency [Hz], PSD [g^2/Hz])], the first one is the default
PRESETS = {
    # profile of visualization/psd_processing.py (psd_input in (m/s^2)^2/Hz with g = 9.81 -> g^2/Hz)
    "Project profile, psd_processing.py (8.1 Grms)": [(20.0, 0.005), (90.0, 0.1), (400.0, 0.1), (2000.0, 0.004)],
    "GEVS qualification (14.1 Grms)": [(20.0, 0.026), (50.0, 0.16), (800.0, 0.16), (2000.0, 0.026)],
    "GEVS acceptance (10.0 Grms)": [(20.0, 0.013), (50.0, 0.08), (800.0, 0.08), (2000.0, 0.013)],
    "NASA workmanship (6.8 Grms)": [(20.0, 0.01), (80.0, 0.04), (500.0, 0.04), (2000.0, 0.01)],
    "Flat 0.01 g2/Hz, 20-2000 Hz (4.4 Grms)": [(20.0, 0.01), (2000.0, 0.01)],
}

# gravity acceleration in the usual unit systems
G_VALUES = (("m/s2 (SI)", 9.80665), ("mm/s2 (mm, t, s)", 9806.65), ("in/s2 (in, lbf, s)", 386.089),
            ("ft/s2", 32.174))


# =====================================================================================================================
#   input PSD
# =====================================================================================================================
def check_psd_table(table):
    """PSD table [(f, W)] -> array (n, 2); f > 0 increasing, W >= 0, at least 2 rows (else ValueError)."""
    table = np.asarray(table, dtype=float).reshape(-1, 2)
    if len(table) < 2:
        raise ValueError("The PSD table needs at least 2 rows")
    if not np.all(np.isfinite(table)):
        raise ValueError("The PSD table contains invalid numbers")
    if np.any(table[:, 0] <= 0) or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError("Frequencies of the PSD table must be > 0 and increasing")
    if np.any(table[:, 1] < 0):
        raise ValueError("PSD values must not be negative")
    return table


def interpolate_psd(table, freq):
    """PSD at frequencies `freq`: log-log interpolation of the table (linear where a value is 0), 0 outside."""
    table = check_psd_table(table)
    f, w = table[:, 0], table[:, 1]
    freq = np.asarray(freq, dtype=float)
    out = np.zeros(freq.shape)
    inside = (freq >= f[0]) & (freq <= f[-1])
    x = freq[inside]
    k = np.clip(np.searchsorted(f, x, side="right") - 1, 0, len(f) - 2)
    f1, f2, w1, w2 = f[k], f[k + 1], w[k], w[k + 1]
    loglog = (w1 > 0) & (w2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.log(w2 / w1) / np.log(f2 / f1)
        value_log = w1 * (x / f1) ** slope
    value_lin = w1 + (w2 - w1) * (x - f1) / (f2 - f1)
    out[inside] = np.where(loglog, value_log, value_lin)
    return out


def psd_rms(table, f_min=None, f_max=None):
    """RMS of a PSD table (exact area of the log-log segments) between f_min and f_max (default: whole table)."""
    table = check_psd_table(table)
    f_min = table[0, 0] if f_min is None else max(f_min, table[0, 0])
    f_max = table[-1, 0] if f_max is None else min(f_max, table[-1, 0])
    if f_max <= f_min:
        return 0.0
    inner = table[(table[:, 0] > f_min) & (table[:, 0] < f_max), 0]
    f = np.concatenate([[f_min], inner, [f_max]])
    w = interpolate_psd(table, f)
    area = 0.0
    for f1, f2, w1, w2 in zip(f[:-1], f[1:], w[:-1], w[1:]):
        if w1 > 0 and w2 > 0:
            n = np.log(w2 / w1) / np.log(f2 / f1)
            if abs(n + 1.0) < 1e-9:
                area += w1 * f1 * np.log(f2 / f1)
            else:
                area += w1 * f1 / (n + 1.0) * ((f2 / f1) ** (n + 1.0) - 1.0)
        else:
            area += 0.5 * (w1 + w2) * (f2 - f1)
    return float(np.sqrt(area))


def trapezoid_weights(freq):
    """Weights of the trapezoidal rule: integral of y df = sum(weights * y)."""
    freq = np.asarray(freq, dtype=float)
    weights = np.zeros(len(freq))
    if len(freq) > 1:
        d = np.diff(freq)
        weights[:-1] += 0.5 * d
        weights[1:] += 0.5 * d
    return weights


def read_psd_csv(path):
    """Reads a PSD table from a text / CSV file: 2 columns frequency, PSD (separator ; , tab or spaces, decimal
    comma allowed with ; or tab); lines which do not start with 2 numbers (header, comments) are skipped."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "$", "!")):
            continue
        if ";" in line or "\t" in line:
            parts = [p.strip().replace(",", ".") for p in re.split(r"[;\t]", line)]
        else:
            parts = [p for p in re.split(r"[,\s]+", line) if p]
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return check_psd_table(rows)


def write_psd_csv(path, table):
    """Saves a PSD table to a CSV file (separator ;)."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Frequency [Hz]", "PSD [g^2/Hz]"])
        for f, w in check_psd_table(table):
            writer.writerow([f"{f:.8g}", f"{w:.8g}"])


# =====================================================================================================================
#   units
# =====================================================================================================================
def material_units_g(h5):
    """Value of g guessed from Young's modulus of MAT1 in the H5 file (SI / mm / inch) -> (g, text) or None."""
    try:
        from .read_Nastran_files import HDF5_LOCK
    except ImportError:
        from file_processing.read_Nastran_files import HDF5_LOCK

    try:
        with HDF5_LOCK:
            h5file = h5._file()
            if "/NASTRAN/INPUT/MATERIAL/MAT1" not in h5file:
                return None
            e = np.asarray(h5file.get_node("/NASTRAN/INPUT/MATERIAL/MAT1").read()["E"], dtype=float)
    except Exception:  # noqa: BLE001
        return None
    e = e[e > 0]
    if not e.size:
        return None
    e = float(np.median(e))
    if e > 1e9:
        return 9.80665, f"E = {e:.3g} -> SI units (m, kg, s)"
    if 1e6 < e < 1e8:
        return 386.089, f"E = {e:.3g} -> inch units (in, lbf, s)"
    if 1e3 < e < 1e6:
        return 9806.65, f"E = {e:.3g} -> mm units (mm, t, s)"
    return None


def guess_g(reference_amplitude=None, h5=None):
    """
    Gravity acceleration in model units -> (value, reason).

    If the amplitude of the enforced acceleration is (about) 1 g in some unit system (9.81, 9810, 386.1, 32.2),
    the input is taken as exactly 1 g (g = that amplitude); otherwise g is guessed from the material, default SI.
    """
    if reference_amplitude is not None:
        amplitude = np.abs(np.asarray(reference_amplitude, dtype=complex))
        amplitude = amplitude[np.isfinite(amplitude) & (amplitude > 0)]
        if amplitude.size:
            a = float(np.median(amplitude))
            for label, g in G_VALUES:
                if abs(a / g - 1.0) < 0.02:
                    return a, f"input amplitude {a:.6g} = 1 g in {label}"
    if h5 is not None:
        guess = material_units_g(h5)
        if guess is not None:
            return guess
    return 9.80665, "SI units assumed"


def von_mises_power(s):
    """|von Mises|^2 of complex stress components (n, 6) X, Y, Z, XY, YZ, ZX by the quadratic form of Segalman."""
    s = np.asarray(s)
    sx, sy, sz, txy, tyz, tzx = (s[..., k] for k in range(6))
    value = (np.abs(sx) ** 2 + np.abs(sy) ** 2 + np.abs(sz) ** 2
             - (sx * np.conj(sy)).real - (sy * np.conj(sz)).real - (sz * np.conj(sx)).real
             + 3.0 * (np.abs(txy) ** 2 + np.abs(tyz) ** 2 + np.abs(tzx) ** 2))
    return np.maximum(value, 0.0)


# =====================================================================================================================
#   random response
# =====================================================================================================================
class RandomResponse:
    """Random response of a frequency response subcase of a ParseH5File (see the module description)."""

    def __init__(self, h5, psd_table, g, reference, step=None):
        """
        Args:
            h5 (ParseH5File): results with a frequency response (complex results)
            psd_table: input acceleration PSD [(f [Hz], W [g^2/Hz])]
            g (float): gravity acceleration in model units (9.80665, 9806.65, 386.089 ...)
            reference: amplitude of the harmonic input acceleration (model units):
                ("node", node ID, component 1 / 2 / 3) - acceleration of the enforced node (complex ACCELERATION
                results are needed), or ("constant", value)
            step (int): a step of the frequency response subcase (default: the first frequency response)
        """
        self.h5 = h5
        self.table = check_psd_table(psd_table)
        self.g = float(g)
        if not self.g > 0:
            raise ValueError("g must be > 0")
        candidates = self.frequency_steps(h5)
        if not candidates:
            raise ValueError("No frequency response (complex results) in the file")
        if step is None or not h5.steps[step]["complex"] or h5.steps[step]["kind"] != "frequency":
            step = candidates[0]
        self.steps = np.array(h5.series_steps(step))
        self.subcase = h5.steps[step]["subcase"]
        self.domains = [h5.steps[i]["domain"] for i in self.steps]
        self.freq = np.array([h5.steps[i]["x"] for i in self.steps], dtype=float)
        self.psd_in = interpolate_psd(self.table, self.freq)
        self.reference = tuple(reference)
        self.ref = self.reference_amplitude(h5, reference, self.domains)
        amplitude = np.abs(self.ref)
        active = self.psd_in > 0
        if not active.any():
            raise ValueError(f"The input PSD ({self.table[0, 0]:.4g} - {self.table[-1, 0]:.4g} Hz) does not cover "
                             f"the frequencies of the frequency response ({self.freq.min():.4g} - "
                             f"{self.freq.max():.4g} Hz)")
        bad = active & ~(np.isfinite(amplitude) & (amplitude > 0))
        if bad.any():
            raise ValueError(f"Reference amplitude is zero / missing at {np.count_nonzero(bad)} frequencies "
                             f"(e.g. {self.freq[bad][0]:.4g} Hz)")
        self.weights = trapezoid_weights(self.freq)
        # |X|^2 -> response PSD [units^2/Hz]
        self.factor = np.zeros(len(self.freq))
        self.factor[active] = self.psd_in[active] * (self.g / amplitude[active]) ** 2
        self.f_range = (float(self.freq[active].min()), float(self.freq[active].max()))
        self.input_rms = float(np.sqrt(np.sum(self.weights * self.psd_in)))   # input Grms at the FRF frequencies
        self.table_rms = psd_rms(self.table)                                 # input Grms of the whole table
        self.table_rms_range = psd_rms(self.table, *self.f_range)            # ... in the range of the response

    # ---------------------------------------------------------------------------------------------- setup
    @staticmethod
    def frequency_steps(h5):
        """First step of every frequency response subcase (complex results)."""
        first = {}
        for i, s in enumerate(getattr(h5, "steps", [])):
            if s["kind"] == "frequency" and s.get("complex") and s.get("family", "") == "":
                first.setdefault(s["series"], i)
        return list(first.values())

    @staticmethod
    def acceleration_table(h5):
        """Key of the complex ACCELERATION table or None."""
        for key, info in h5._tables.items():
            if info["kind"] == "nodal" and info["complex"] and info["base"] == "ACCELERATION":
                return key
        return None

    @classmethod
    def reference_amplitude(cls, h5, reference, domains):
        """Complex amplitude of the input acceleration at the domains (frequencies)."""
        if reference[0] == "constant":
            value = float(reference[1])
            if value == 0:
                raise ValueError("The reference amplitude must not be 0")
            return np.full(len(domains), value, dtype=complex)
        node_id, component = int(reference[1]), int(reference[2])
        index = int(h5.node_index([node_id])[0])
        if index < 0:
            raise ValueError(f"Reference node {node_id} is not in the model")
        key = cls.acceleration_table(h5)
        if key is None:
            raise ValueError("No complex ACCELERATION results - use a constant reference amplitude")
        comps = h5._tables[key]["comps"]
        name = "XYZ"[component - 1]
        if name not in comps:
            raise ValueError(f"Component {name} is not in the ACCELERATION results")
        data = h5.nodal_history(key, index, domains)[:, comps.index(name)]
        if not np.isfinite(data).any():
            raise ValueError(f"Acceleration of node {node_id} is not in the results (output set?)")
        return data

    # ---------------------------------------------------------------------------------------------- sources
    def is_acceleration(self, source):
        return source[0] == "nodal" and self.h5._tables[source[1]]["base"] == "ACCELERATION"

    def unit(self, source):
        """Unit of the response: "g" for accelerations, "" = model units."""
        return "g" if self.is_acceleration(source) else ""

    def map_sources(self):
        """Result groups which can be converted to RMS maps -> [(source, label)]."""
        out = []
        first = self.domains[0]
        for key, info in self.h5._tables.items():
            if info["kind"] == "nodal" and info["complex"] and first in info["blocks"]:
                label = self.h5._nodal_label(info["base"]) + (" [g]" if info["base"] == "ACCELERATION" else "")
                out.append((("nodal", key), label))
        for base in ("STRESS", "STRAIN"):
            if any(i["kind"] == "element" and i["base"] == base and i["complex"] and first in i["blocks"]
                   for i in self.h5._tables.values()):
                out.append((("element", base), f"{base.capitalize()} (element + nodal avg.)"))
        return out

    def source_of_quantity(self, name):
        """Source of the PSD of a complex result of the viewer (e.g. "Acceleration Y") or None."""
        q = self.h5._quantities.get(name)
        if q is None or not self.h5.is_complex(name):
            return None
        if q["source"] == "nodal":
            if self.domains[0] not in self.h5._tables[q["table"]]["blocks"]:
                return None
            return ("nodal", q["table"], tuple(q["cols"]))
        if q["op"] == "component":
            return ("element", q["base"], q["where"], ("component", q["col"]))
        return None

    # ---------------------------------------------------------------------------------------------- PSD, RMS
    def psd(self, source, index):
        """Response PSD (n_freq,) of a source in node / element `index` [unit^2/Hz] (NaN = no result).

        source: ("nodal", table key, columns) - sum of |X|^2 of the columns (1 column = component, more = magnitude)
                ("element", base, "cell" / "node", ("component", k) or "von_mises")
        """
        if source[0] == "nodal":
            _, key, cols = source
            data = self.h5.nodal_history(key, index, self.domains)[:, list(cols)]
            power = np.sum(np.abs(data) ** 2, axis=1)
            power[~np.isfinite(data).all(axis=1)] = np.nan
            if self.is_acceleration(source):
                power = power / self.g ** 2
        else:
            _, base, where, op = source
            stress = self.h5._element_history(dict(complex=True, base=base, where=where), index, self.domains)
            power = von_mises_power(stress) if op == "von_mises" else np.abs(stress[:, op[1]]) ** 2
            power = np.where(np.isfinite(stress).all(axis=1), power, np.nan)
        return self.factor * power

    def rms(self, psd):
        """RMS of a response PSD (trapezoidal rule over the frequencies of the frequency response)."""
        psd = np.asarray(psd, dtype=float)
        ok = np.isfinite(psd)
        return float(np.sqrt(np.sum(self.weights[ok] * psd[ok]))) if ok.any() else float("nan")

    def input_curve(self, n=400):
        """Input PSD for a graph: (f, W) in the frequency range of the response."""
        f = np.geomspace(self.f_range[0], self.f_range[1], n) if self.f_range[1] > self.f_range[0] \
            else np.array(self.f_range)
        f = np.unique(np.concatenate([f, self.table[(self.table[:, 0] >= self.f_range[0])
                                                    & (self.table[:, 0] <= self.f_range[1]), 0]]))
        return f, interpolate_psd(self.table, f)

    # ---------------------------------------------------------------------------------------------- RMS maps
    def compute_maps(self, sources, progress=None):
        """
        RMS (1 sigma) of all nodes / elements for the sources [("nodal", table key) / ("element", "STRESS")].

        Returns: list of dicts {name, location, group, item, values, source, unit}
        """
        h5 = self.h5
        report = progress or (lambda fraction, text: None)
        coef = self.factor * self.weights
        active = np.flatnonzero(coef > 0)
        total = max(1, len(active) * len(sources))
        done = 0
        maps = []
        for source in sources:
            if source[0] == "nodal":
                key = source[1]
                info = h5._tables[key]
                label = h5._nodal_label(info["base"])
                acceleration = info["base"] == "ACCELERATION"
                unit = " [g]" if acceleration else ""
                total_power = np.zeros((len(h5.node_ids), len(info["comps"])))
                found = np.zeros(len(h5.node_ids), dtype=bool)
                for i in active:
                    report(done / total, f"random RMS: {label} {self.freq[i]:.4g} Hz")
                    done += 1
                    array = h5._nodal_array(key, self.domains[i], cache=False)
                    if array is None:
                        continue
                    ok = np.isfinite(array).all(axis=1)
                    found |= ok
                    total_power[ok] += coef[i] * np.abs(array[ok]) ** 2
                if acceleration:
                    total_power /= self.g ** 2
                total_power[~found] = np.nan
                comps = info["comps"]
                for group_comps, magnitude in ((("X", "Y", "Z"), "magnitude"), (("RX", "RY", "RZ"), "rot. magnitude")):
                    cols = [comps.index(c) for c in group_comps if c in comps]
                    if not cols:
                        continue
                    if magnitude.startswith("rot") and not np.nan_to_num(total_power[:, cols]).any():
                        continue                                # rotations of a solid model are zero
                    items = [(magnitude, cols)] + [(comps[c], [c]) for c in cols]
                    for item, item_cols in items:
                        maps.append(dict(name=f"RMS {label} {item}{unit}", location="point", group=f"{label}{unit}",
                                         item=item, values=np.sqrt(total_power[:, item_cols].sum(axis=1)),
                                         source=("nodal", key, tuple(item_cols)), unit=unit.strip(" []")))
            else:
                base = source[1]
                label = base.capitalize()
                n = {"cell": len(h5.element_ids), "node": len(h5.node_ids)}
                comp_power = {w: np.zeros((n[w], 6)) for w in n}
                vm_power = {w: np.zeros(n[w]) for w in n}
                found = {w: np.zeros(n[w], dtype=bool) for w in n}
                for i in active:
                    report(done / total, f"random RMS: {label} {self.freq[i]:.4g} Hz")
                    done += 1
                    arrays = h5._element_arrays(base, True, self.domains[i], cache=False)
                    if arrays is None:
                        continue
                    for where in n:
                        s = arrays[where]
                        ok = np.isfinite(s).all(axis=1)
                        found[where] |= ok
                        comp_power[where][ok] += coef[i] * np.abs(s[ok]) ** 2
                        if base == "STRESS":
                            vm_power[where][ok] += coef[i] * von_mises_power(s[ok])
                for where, location, suffix in (("cell", "cell", "(element)"), ("node", "point", "(nodal avg.)")):
                    comp_power[where][~found[where]] = np.nan
                    vm_power[where][~found[where]] = np.nan
                    group = f"{label} {suffix}"
                    items = [("von Mises", "von_mises")] if base == "STRESS" else []
                    items += [(c, ("component", k)) for k, c in enumerate(("XX", "YY", "ZZ", "XY", "YZ", "ZX"))]
                    for item, op in items:
                        values = np.sqrt(vm_power[where] if op == "von_mises" else comp_power[where][:, op[1]])
                        maps.append(dict(name=f"RMS {label} {item} {suffix}", location=location, group=group,
                                         item=item, values=values, source=("element", base, where, op), unit=""))
        report(1.0, "random RMS: done")
        return maps
