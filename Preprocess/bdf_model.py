"""
Preprocessing - reading, analysis and checks of a Nastran input file (*.bdf, *.dat, *.nas).

BdfModel(file) reads the model by pyNastran and prepares:
    * mesh for the 3D view (same machinery as the H5 results: file_processing.read_Nastran_files.NastranMesh)
    * model fields shown in the viewer: Property ID, Material ID, element type, connected body, contact body
      (BSURF), element volume, element mass, element quality (scaled Jacobian, edge ratio)
    * model entities for the 3D view: SPC / SPCD nodes, rigid elements, concentrated masses, gravity, forces
    * text report (self.report): summary, materials and properties, mass and CG, case control, constraints, loads,
      dynamic cards, rigid elements, contact, mesh checks (connected bodies, coincident / unused nodes, quality,
      references from the case control)

Example:
    model = BdfModel("model.bdf")
    print(model.report)
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter, vtkMeshQuality

ROOT = Path(__file__).resolve().parents[1]                      # project root -> file_processing package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from file_processing.read_Nastran_files import NastranMesh  # noqa: E402

SOLID_TYPES = {10, 12, 13, 14, 24, 25, 26, 27}                  # VTK 3D cell types
SHELL_TYPES = {5, 9, 22, 23}                                    # VTK 2D cell types
LINE_TYPES = {3}


def _fmt(value, digits=4):
    return f"{value:.{digits}g}" if isinstance(value, (int, float, np.floating, np.integer)) else str(value)


def _vtk_name(vtk_type):
    return NastranMesh._VTK_NAMES.get(int(vtk_type), str(vtk_type))


def _table(header, rows):
    """Simple text table (list of rows) -> list of lines."""
    cells = [list(map(str, header))] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  " + "  ".join(c.ljust(w) for c, w in zip(cells[0], widths)),
             "  " + "  ".join("-" * w for w in widths)]
    lines += ["  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells[1:]]
    return lines


class BdfModel(NastranMesh):
    # limits for the element quality check in the report
    MIN_SCALED_JACOBIAN = 0.2
    MAX_EDGE_RATIO = 10.0

    def __init__(self, file_path, verbose=True, coincident_tolerance=None, progress=None):
        """
        Args:
            file_path: Nastran input file (with or without executive / case control)
            verbose (bool): print progress
            coincident_tolerance (float): distance of coincident nodes, default 1e-6 x model size
            progress: optional function progress(fraction 0..1 or None = unknown, text) called during loading
                (background loading in the viewer)
        """
        super().__init__(file_path, verbose)
        from pyNastran.bdf.bdf import BDF

        report = progress or (lambda fraction, text: None)
        self.results_title = "model fields"
        self.coincident_tolerance = coincident_tolerance
        t0 = time.time()
        self._log(f"Reading {self.file_path.name} by pyNastran ...")
        report(None, "reading the file by pyNastran")
        self.bdf = BDF(debug=None)                                  # debug=None -> warnings and errors only
        self._read_bdf()
        self.read_time = time.time() - t0

        self._log("Analysing the model ...")
        self.checks = []                                            # (level, text) - errors / warnings / info
        report(0.55, "nodes and elements")
        self._read_nodes()
        self._read_elements()
        report(0.65, "model fields (volume, mass, quality, bodies)")
        self.to_pyvista()                                           # grid + model fields
        report(0.8, "model entities")
        self._collect_entities()
        report(0.85, "report")
        self.report = self._make_report()
        self._log(f"Model {self.file_path.name} was analysed ({time.time() - t0:.1f} s)")

    # ================================================================================================= reading
    def _read_bdf(self):
        try:
            self.bdf.read_bdf(str(self.file_path), xref=False)
        except Exception as error:                                  # e.g. bulk data only -> punch=True
            self._log(f"  standard read failed ({error}), trying bulk data only (punch=True) ...")
            from pyNastran.bdf.bdf import BDF
            self.bdf = BDF(debug=None)
            try:
                self.bdf.read_bdf(str(self.file_path), xref=False, punch=True)
            except Exception:
                raise error

    def _read_nodes(self):
        nodes = self.bdf.nodes
        nids = np.array(sorted(nodes), dtype=np.int64)
        xyz = np.array([nodes[n].xyz for n in nids], dtype=np.float64)
        cp = np.array([nodes[n].cp for n in nids])
        cd = np.array([nodes[n].cd for n in nids])
        if np.any(cp != 0):
            try:                                                    # transform to the basic coordinate system
                xyz_basic = self.bdf.get_xyz_in_coord(cid=0, fdtype="float64")
                if xyz_basic.shape == xyz.shape:
                    xyz = xyz_basic
                self.checks.append(("info", f"{np.count_nonzero(cp)} GRIDs defined in local systems (CP != 0) "
                                            f"- transformed to the basic system"))
            except Exception as error:
                self.checks.append(("warning", f"{np.count_nonzero(cp)} GRIDs have CP != 0 and could not be "
                                               f"transformed ({error}) - shown in local coordinates!"))
        self._set_nodes(nids, xyz)
        self.node_cd = cd

    def _read_elements(self):
        by_type = defaultdict(list)
        for elem in self.bdf.elements.values():
            by_type[elem.type].append(elem)
        for name, elems in sorted(by_type.items()):
            node_lists = [[n or 0 for n in elem.node_ids] for elem in elems]
            width = max(len(n) for n in node_lists)
            nodes = np.zeros((len(elems), width), dtype=np.int64)
            for i, n in enumerate(node_lists):
                nodes[i, :len(n)] = n
            pids = [self._int(getattr(elem, "pid", 0)) for elem in elems]
            self._add_elements(name, [elem.eid for elem in elems], pids, nodes)
        self._finish_elements()

    @staticmethod
    def _int(value):
        """ID from a pyNastran field (int / object with an ID / None)."""
        if value is None:
            return 0
        if isinstance(value, (int, np.integer)):
            return int(value)
        for attr in ("pid", "mid", "nid", "cid", "eid"):
            if hasattr(value, attr):
                return int(getattr(value, attr))
        return 0

    # ================================================================================================= fields
    def to_pyvista(self, linear=False):
        if self._grid is None:
            grid = super().to_pyvista()                             # mesh + "Property ID"
            self._add_fields(grid)
        return super().to_pyvista(linear)

    def _material_of_property(self, pid):
        prop = self.bdf.properties.get(pid)
        if prop is None:
            return 0
        for attr in ("mid", "mid1"):
            mid = self._int(getattr(prop, attr, None))
            if mid:
                return mid
        mids = getattr(prop, "mids", None)                          # PCOMP
        return self._int(mids[0]) if mids else 0

    def _add_fields(self, grid):
        self.results = []
        self.result_meta = {}

        def add(name, values, group, item, categorical=False, annotations=None):
            grid.cell_data[name] = values
            self._grid_values[name] = np.asarray(values, dtype=float)   # values() without VTK (threads)
            self._register(name, "cell", group, item)
            if categorical:
                self.categorical_results.add(name)
            if annotations:
                self.result_annotations[name] = annotations

        # ---- materials of elements
        mid_of_pid = {pid: self._material_of_property(pid) for pid in np.unique(self.property_ids)}
        self.material_ids = np.array([mid_of_pid.get(p, 0) for p in self.property_ids], dtype=np.int64)
        conrod = self.element_types == "CONROD"                     # CONROD has a material instead of a property
        if conrod.any():
            mids = {e.eid: self._int(e.mid) for e in self.bdf.elements.values() if e.type == "CONROD"}
            self.material_ids[conrod] = [mids.get(e, 0) for e in self.element_ids[conrod]]

        # ---- element type codes (e.g. CTETRA (TETRA10))
        type_names = [f"{t} ({_vtk_name(c)})" for t, c in zip(self.element_types, grid.celltypes)]
        self.type_labels = sorted(set(type_names))
        codes = {name: i for i, name in enumerate(self.type_labels)}
        type_codes = np.array([codes[n] for n in type_names], dtype=np.int64)

        # ---- size, mass
        size_data = self._cell_sizes()
        celltypes = np.asarray(grid.celltypes)
        is_solid = np.isin(celltypes, list(SOLID_TYPES))
        is_shell = np.isin(celltypes, list(SHELL_TYPES))
        is_line = np.isin(celltypes, list(LINE_TYPES))
        # abs(): sign of the volume differs between VTK versions for wedges, inverted elements -> scaled Jacobian
        self.volumes = np.where(is_solid, np.abs(np.asarray(size_data["Volume"])), np.nan)
        self.areas = np.where(is_shell, np.asarray(size_data["Area"]), np.nan)
        self.lengths = np.where(is_line, np.asarray(size_data["Length"]), np.nan)
        self.centers = np.asarray(grid.cell_centers().points)
        self.element_mass = self._element_masses(is_solid, is_shell, is_line)

        # ---- connected bodies, contact bodies, quality
        self.body_ids = self._connected_bodies(grid)
        self.contact_body_ids = self._contact_bodies()
        linear = self._make_grid(self._linear_blocks(), self.points)
        self.scaled_jacobian = self._quality(linear, "ScaledJacobian", {10, 12, 13, 14, 5, 9})
        self.edge_ratio = self._quality(linear, "EdgeRatio", {10, 12, 13, 5, 9})

        prop_id = np.asarray(grid.cell_data["Property ID"])
        add("Property ID", prop_id, "Model", "Property ID", categorical=True)
        add("Material ID", self.material_ids, "Model", "Material ID", categorical=True)
        add("Element type", type_codes, "Model", "Element type", categorical=True,
            annotations={i: name for i, name in enumerate(self.type_labels)})
        add("Body (connected mesh part)", self.body_ids, "Model", "Body (connected mesh part)", categorical=True)
        if np.any(self.contact_body_ids):
            add("Contact body (BSURF ID)", self.contact_body_ids, "Model", "Contact body (BSURF ID)",
                categorical=True)
        if is_solid.any():
            add("Element volume", self.volumes, "Size and mass", "Volume")
        if is_shell.any():
            add("Element area", self.areas, "Size and mass", "Area")
        add("Element mass", self.element_mass, "Size and mass", "Mass")
        add("Scaled Jacobian", self.scaled_jacobian, "Element quality", "Scaled Jacobian (1 = ideal)")
        add("Edge ratio", self.edge_ratio, "Element quality", "Edge ratio (longest / shortest edge)")

    def _cell_sizes(self, chunk=10000):
        """Volume / area / length of all cells (vtkCellSizeFilter, exact also for quadratic elements). Computed in
        chunks of cells - a long VTK call blocks other Python threads (the window while loading in background)."""
        out = {"Volume": [], "Area": [], "Length": []}
        for vtk_type, conn in self._cell_blocks:
            for start in range(0, len(conn), chunk):
                sub = self._make_grid([(vtk_type, conn[start:start + chunk])], self.points)
                sizes = vtkCellSizeFilter()
                sizes.SetInputData(sub)
                sizes.ComputeVertexCountOff()
                sizes.ComputeSumOff()
                sizes.Update()
                data = pv.wrap(sizes.GetOutput()).cell_data
                for key in out:
                    out[key].append(np.asarray(data[key], dtype=float).copy())
                time.sleep(0)                                       # let the window thread run
        return {key: np.concatenate(values) if values else np.zeros(0) for key, values in out.items()}

    def _element_masses(self, is_solid, is_shell, is_line):
        rho = {mid: float(getattr(mat, "rho", 0.0) or 0.0) for mid, mat in self.bdf.materials.items()}
        mass = np.full(len(self.element_ids), np.nan)
        element_rho = np.array([rho.get(m, np.nan) for m in self.material_ids])
        mass[is_solid] = element_rho[is_solid] * self.volumes[is_solid]

        unsupported = Counter()
        for i in np.flatnonzero(is_shell | is_line):
            prop = self.bdf.properties.get(int(self.property_ids[i]))
            ptype = getattr(prop, "type", None)
            if is_shell[i] and ptype == "PSHELL":
                mass[i] = self.areas[i] * (element_rho[i] * float(prop.t or 0.0) + float(prop.nsm or 0.0))
            elif is_line[i] and ptype in ("PROD", "PBAR"):
                mass[i] = self.lengths[i] * (element_rho[i] * float(prop.A or 0.0) + float(prop.nsm or 0.0))
            else:
                unsupported[f"{self.element_types[i]} / {ptype}"] += 1
        for key, count in unsupported.items():
            self.checks.append(("info", f"mass of {count} elements {key} not calculated (unsupported property)"))
        missing = np.isnan(element_rho) & is_solid
        if missing.any():
            self.checks.append(("error", f"{np.count_nonzero(missing)} solid elements without a valid property / "
                                         f"material (no density)"))
        return mass

    def _connected_bodies(self, grid):
        """Connected parts of the mesh (elements sharing nodes), numbered by size: 1 = the largest one."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n_cells, n_points = grid.n_cells, grid.n_points
        offsets = np.asarray(grid.offset)
        rows = np.repeat(np.arange(n_cells), np.diff(offsets))
        cols = n_cells + np.asarray(grid.cell_connectivity)
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells + n_points, n_cells + n_points))
        _, labels = connected_components(graph, directed=False)
        cell_labels = labels[:n_cells]
        unique, counts = np.unique(cell_labels, return_counts=True)
        order = unique[np.argsort(-counts, kind="stable")]
        renumber = np.zeros(labels.max() + 1, dtype=np.int64)
        renumber[order] = np.arange(1, len(order) + 1)
        return renumber[cell_labels]

    def _contact_bodies(self):
        """Contact body (BSURF ID) of elements, 0 = element not in any BSURF."""
        result = np.zeros(len(self.element_ids), dtype=np.int64)
        for sid, bsurf in sorted(getattr(self.bdf, "bsurf", {}).items()):
            idx = self.cell_index(bsurf.eids)
            result[idx[idx >= 0]] = sid
            if np.any(idx < 0):
                self.checks.append(("warning", f"BSURF {sid}: {np.count_nonzero(idx < 0)} unknown elements"))
        return result

    @staticmethod
    def _quality(linear_grid, measure, supported_types):
        quality = vtkMeshQuality()
        quality.SetInputData(linear_grid)
        for cell in ("Tet", "Hex", "Wedge", "Pyramid", "Triangle", "Quad"):
            setter = getattr(quality, f"Set{cell}QualityMeasureTo{measure}", None)
            if setter is not None:
                setter()
        quality.Update()
        values = np.asarray(pv.wrap(quality.GetOutput()).cell_data["Quality"], dtype=float).copy()
        values[~np.isin(np.asarray(linear_grid.celltypes), list(supported_types))] = np.nan
        return values

    # ================================================================================================= entities
    def _node_xyz(self, node_ids):
        idx = self.node_index(np.asarray(node_ids, dtype=np.int64))
        return self.points[idx[idx >= 0]], idx

    def _collect_entities(self):
        """Model entities for the 3D view: {key: dict(kind, label, color, points, lines / vectors, texts)}"""
        bdf = self.bdf
        self.entities = {}
        size = float(np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0)))

        # ---- SPC (all sets) + permanent constraints on GRIDs (PS)
        spc_nodes = defaultdict(set)                                # node ID -> constrained DOFs
        for sid, cards in bdf.spcs.items():
            for card in cards:
                components = card.components
                for i, nid in enumerate(card.nodes):
                    comp = components[i] if isinstance(components, list) else components
                    spc_nodes[int(nid)].update(str(comp))
        for nid, node in bdf.nodes.items():
            if getattr(node, "ps", ""):
                spc_nodes[int(nid)].update(str(node.ps))
        self.spc_nodes = {nid: "".join(sorted(d)) for nid, d in spc_nodes.items()}
        if self.spc_nodes:
            xyz, _ = self._node_xyz(list(self.spc_nodes))
            texts = [f"SPC {d}" for d in self.spc_nodes.values()] if len(xyz) <= 30 else None
            self.entities["spc"] = dict(kind="points", label="SPC constraints", color="orange", points=xyz,
                                        texts=texts, count=len(xyz))

        # ---- SPCD (enforced motion) + forces / moments / gravity from the loads
        spcd, forces, moments, gravity = [], [], [], []
        for sid, cards in bdf.loads.items():
            for card in cards:
                if card.type == "SPCD":
                    for nid, comp, value in zip(card.nodes, card.components, card.enforced):
                        spcd.append((int(nid), comp, value, sid))
                elif card.type in ("FORCE", "MOMENT"):
                    vector = np.asarray(card.xyz, dtype=float) * float(card.mag)
                    (forces if card.type == "FORCE" else moments).append((int(card.node), vector, sid, card.cid))
                elif card.type == "GRAV":
                    gravity.append((np.asarray(card.N, dtype=float) * float(card.scale), sid, card.cid))
        self.spcd = spcd
        if spcd:
            xyz, _ = self._node_xyz([s[0] for s in spcd])
            texts = [f"SPCD {s[3]}: comp {s[1]} = {_fmt(s[2])}" for s in spcd] if len(spcd) <= 30 else None
            self.entities["spcd"] = dict(kind="points", label="SPCD enforced motion", color="red", points=xyz,
                                         texts=texts, count=len(xyz))
        for key, items, label, color in (("force", forces, "FORCE", "blue"), ("moment", moments, "MOMENT", "cyan")):
            if items:
                xyz, idx = self._node_xyz([f[0] for f in items])
                vectors = np.array([f[1] for f in items])[idx >= 0]
                magnitude = np.linalg.norm(vectors, axis=1)
                scale = 0.08 * size / magnitude.max() if magnitude.max() > 0 else 0.0
                self.entities[key] = dict(kind="arrows", label=label, color=color, points=xyz,
                                          vectors=vectors * scale, count=len(xyz))
        self.gravity = gravity
        if gravity:
            top = np.array([self.points[:, 0].mean(), self.points[:, 1].mean(), self.points[:, 2].max()])
            vectors, texts, starts = [], [], []
            for i, (g, sid, cid) in enumerate(gravity):
                direction = g / np.linalg.norm(g) if np.linalg.norm(g) > 0 else np.array([0.0, 0.0, -1.0])
                starts.append(top - direction * 0.3 * size + i * 0.05 * size)
                vectors.append(direction * 0.25 * size)
                texts.append(f"GRAV {sid}: ({_fmt(g[0])}, {_fmt(g[1])}, {_fmt(g[2])})"
                             + (f" in CID {cid}" if cid else ""))
            self.entities["grav"] = dict(kind="arrows", label="GRAV gravity", color="green",
                                         points=np.array(starts), vectors=np.array(vectors), texts=texts,
                                         count=len(gravity))

        # ---- rigid elements (spider: center node -> other nodes)
        segments, centers = [], []
        self.rigid_summary = Counter()
        for elem in bdf.rigid_elements.values():
            self.rigid_summary[elem.type] += 1
            independent = [int(n) for n in elem.independent_nodes if n]
            dependent = [int(n) for n in elem.dependent_nodes if n]
            if len(independent) == 1:
                center, others = independent[0], dependent
            elif dependent:
                center, others = dependent[0], independent + dependent[1:]
            else:
                continue
            centers.append(center)
            segments += [(center, other) for other in others]
        if segments:
            pairs = self.node_index(np.array(segments, dtype=np.int64))
            pairs = pairs[(pairs >= 0).all(axis=1)]
            xyz, _ = self._node_xyz(centers)
            self.entities["rigid"] = dict(kind="lines", label="Rigid elements (" + ", ".join(
                f"{k} {v}" for k, v in self.rigid_summary.items()) + ")", color="magenta", points=xyz,
                lines=self.points[pairs], count=sum(self.rigid_summary.values()))

        # ---- concentrated masses
        self.mass_points = []                                       # (mass, position, card)
        for mass in bdf.masses.values():
            if mass.type == "CONM2":
                nid = int(self._int(mass.nid))
                xyz, idx = self._node_xyz([nid])
                if not len(xyz):
                    continue
                offset = np.asarray(mass.X, dtype=float)
                cid = self._int(mass.cid)
                if cid == -1:
                    position = offset
                else:
                    position = xyz[0] + offset
                    if cid > 0 and np.any(offset):
                        self.checks.append(("info", f"CONM2 {mass.eid}: offset in CID {cid} used as basic"))
                self.mass_points.append((float(mass.mass), position, mass))
            else:
                self.checks.append(("info", f"{mass.type} {mass.eid}: mass not included in the mass summary"))
        if self.mass_points:
            texts = [f"CONM2 {m[2].eid}: {_fmt(m[0])}" for m in self.mass_points] if len(self.mass_points) <= 30 \
                else None
            self.entities["mass"] = dict(kind="points", label="CONM2 masses", color="black",
                                         points=np.array([m[1] for m in self.mass_points]), texts=texts,
                                         count=len(self.mass_points))

    # ================================================================================================= report
    def _units_guess(self):
        e_values = [float(m.e) for m in self.bdf.materials.values() if m.type == "MAT1" and m.e]
        rho_values = [float(m.rho) for m in self.bdf.materials.values() if getattr(m, "rho", 0)]
        if not e_values:
            return "unknown (no MAT1 with E)"
        e, rho = np.median(e_values), np.median(rho_values) if rho_values else 0.0
        if e > 1e8:
            return "SI: m, kg, s, N, Pa  (E ~ %.3g, rho ~ %.3g)" % (e, rho)
        if 1e3 < e < 1e6 and 0 < rho < 1e-5:
            return "mm, t, s, N, MPa  (E ~ %.3g, rho ~ %.3g)" % (e, rho)
        if 1e5 < e < 1e8 and 0 < rho < 1e-2:
            return "in, lbf, s, psi  (E ~ %.3g, rho ~ %.3g)" % (e, rho)
        return "unclear (E ~ %.3g, rho ~ %.3g)" % (e, rho)

    def _mass_properties(self):
        solid = np.isfinite(self.element_mass)
        m_el = self.element_mass[solid]
        x_el = self.centers[solid]
        m_pt = np.array([m[0] for m in self.mass_points]) if self.mass_points else np.zeros(0)
        x_pt = np.array([m[1] for m in self.mass_points]) if self.mass_points else np.zeros((0, 3))
        masses = np.concatenate([m_el, m_pt])
        positions = np.vstack([x_el, x_pt])
        total = masses.sum()
        if total <= 0:
            return 0.0, np.zeros(3), np.zeros((3, 3)), m_el.sum(), m_pt.sum()
        cg = (masses[:, None] * positions).sum(axis=0) / total
        r = positions - cg
        inertia = np.einsum("i,ij,ik->jk", masses, r, r)
        inertia = np.trace(inertia) * np.eye(3) - inertia
        return total, cg, inertia, m_el.sum(), m_pt.sum()

    def _case_control_checks(self):
        """References of the case control commands to the bulk data."""
        bdf = self.bdf
        ccd = bdf.case_control_deck
        if ccd is None:
            return []
        refs = {"SPC": ("spcadds", "spcs"), "MPC": ("mpcadds", "mpcs"), "LOAD": ("load_combinations", "loads"),
                "DLOAD": ("dloads", "dload_entries"), "METHOD": ("methods",), "CMETHOD": ("cMethods",),
                "FREQUENCY": ("frequencies",), "SDAMPING": ("tables_sdamping",), "LOADSET": ("load_combinations",),
                "TSTEP": ("tsteps",), "NLPARM": ("nlparms",), "TSTEPNL": ("tstepnls",)}
        out = []
        for sc_id, subcase in sorted(ccd.subcases.items()):
            for key, attrs in refs.items():
                if key not in subcase.params:
                    continue
                value = subcase.params[key][0]
                if not isinstance(value, (int, np.integer)):
                    continue
                defined = any(value in getattr(bdf, a, {}) for a in attrs)
                where = "global" if sc_id == 0 else f"SUBCASE {sc_id}"
                if defined:
                    out.append(("ok", f"{where}: {key} = {value} found"))
                else:
                    out.append(("error", f"{where}: {key} = {value} is NOT defined in the bulk data"))
        # SPCADD / LSEQ references
        for sid, cards in bdf.spcadds.items():
            for card in cards:
                for s in card.sets:
                    if self._int(s) not in bdf.spcs:
                        out.append(("error", f"SPCADD {sid}: SPC set {s} is NOT defined"))
        for sid, cards in bdf.load_combinations.items():
            for card in cards:
                if card.type == "LSEQ" and self._int(card.lid) and self._int(card.lid) not in bdf.loads:
                    out.append(("error", f"LSEQ {sid}: load set {card.lid} is NOT defined"))
        return out

    def _make_report(self):
        bdf = self.bdf
        L = []

        def section(title):
            L.extend(["", "=" * 100, title, "=" * 100])

        # ---- summary
        section("MODEL SUMMARY")
        L.append(f"file:            {self.file_path}")
        L.append(f"size:            {self.file_path.stat().st_size / 1e6:.1f} MB, read in {self.read_time:.1f} s")
        L.append(f"solution:        SOL {bdf.sol}" if bdf.sol else "solution:        - (no executive control)")
        L.append(f"units (guess):   {self._units_guess()}")
        L.append(f"nodes:           {len(self.node_ids)}")
        L.append(f"elements:        {len(self.element_ids)} shown in 3D")
        for key, count in sorted(self.element_counts.items()):
            L.append(f"                   {key:28s} {count}")
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        L.append(f"bounding box:    X {_fmt(lo[0])} .. {_fmt(hi[0])}, Y {_fmt(lo[1])} .. {_fmt(hi[1])}, "
                 f"Z {_fmt(lo[2])} .. {_fmt(hi[2])}")
        L.append(f"model size:      {_fmt(float(np.linalg.norm(hi - lo)))} (diagonal)")
        if bdf.coords and len(bdf.coords) > 1:
            L.append(f"coord. systems:  {sorted(c for c in bdf.coords if c != 0)}")

        # ---- materials, properties
        section("MATERIALS")
        rows = []
        for mid, mat in sorted(bdf.materials.items()):
            n_el = int(np.count_nonzero(self.material_ids == mid))
            mass = np.nansum(self.element_mass[self.material_ids == mid])
            name = self._comment_name(mat)
            if mat.type == "MAT1":
                rows.append([mid, mat.type, _fmt(mat.e), _fmt(mat.nu), _fmt(mat.rho), n_el, _fmt(mass), name])
            else:
                rows.append([mid, mat.type, "-", "-", _fmt(getattr(mat, "rho", "-")), n_el, _fmt(mass), name])
        L += _table(["MID", "type", "E", "nu", "rho", "elements", "mass", "name"], rows)
        for mid, mat in sorted(bdf.materials.items()):
            if mat.type == "MAT1":
                if not mat.e or mat.e <= 0:
                    self.checks.append(("error", f"MAT1 {mid}: E = {mat.e}"))
                if mat.nu is not None and not (0.0 <= mat.nu < 0.5):
                    self.checks.append(("warning", f"MAT1 {mid}: nu = {mat.nu}"))
                if not mat.rho:
                    self.checks.append(("warning", f"MAT1 {mid}: rho = {mat.rho} (no mass)"))

        section("PROPERTIES")
        rows = []
        for pid, prop in sorted(bdf.properties.items()):
            sel = self.property_ids == pid
            types = ", ".join(sorted(set(self.element_types[sel])))
            rows.append([pid, prop.type, self._material_of_property(pid), int(np.count_nonzero(sel)), types,
                         _fmt(np.nansum(self.volumes[sel])) if np.isfinite(self.volumes[sel]).any() else "-",
                         _fmt(np.nansum(self.element_mass[sel])) if np.any(sel) else "-", self._comment_name(prop)])
            if not np.any(sel):
                self.checks.append(("info", f"{prop.type} {pid} is not used by any element"))
        L += _table(["PID", "type", "MID", "elements", "element types", "volume", "mass", "name"], rows)
        missing = sorted(set(self.property_ids[self.element_types != "CONROD"].tolist()) - set(bdf.properties))
        if missing:
            self.checks.append(("error", f"elements reference missing properties: {missing[:20]}"))

        # ---- mass
        section("MASS PROPERTIES (elements: density x volume / area x thickness; CONM2 as point masses)")
        total, cg, inertia, m_el, m_pt = self._mass_properties()
        L.append(f"structural mass: {_fmt(m_el, 6)}")
        L.append(f"CONM2 masses:    {_fmt(m_pt, 6)}  ({len(self.mass_points)} x)")
        L.append(f"total mass:      {_fmt(total, 6)}")
        L.append(f"center of mass:  ({_fmt(cg[0], 6)}, {_fmt(cg[1], 6)}, {_fmt(cg[2], 6)})")
        L.append("inertia at CG (approx., element masses at centroids):")
        for row in inertia:
            L.append("                 " + "  ".join(f"{v:12.5e}" for v in row))
        for g, sid, cid in self.gravity:
            L.append(f"weight in GRAV {sid}: {_fmt(total * np.linalg.norm(g), 6)}  (= total mass x |GRAV|)")

        # ---- case control
        section("CASE CONTROL")
        if bdf.case_control_deck is not None:
            text = str(bdf.case_control_deck).replace("BEGIN BULK", "").rstrip()
            L += ["  " + line for line in text.splitlines()]
            checks = self._case_control_checks()
            L.append("")
            L += [f"  [{level.upper()}] {text}" for level, text in checks]
            self.checks += [c for c in checks if c[0] != "ok"]
        else:
            L.append("  - (bulk data only)")

        # ---- constraints, loads, dynamic cards
        section("CONSTRAINTS AND LOADS")
        for sid, cards in sorted(bdf.spcadds.items()):
            L += [f"  SPCADD {sid}: sets {[self._int(s) for s in c.sets]}" for c in cards]
        for sid, cards in sorted(bdf.spcs.items()):
            for card in cards:
                nodes = list(card.nodes)
                L.append(f"  {card.type} {sid}: {len(nodes)} nodes, DOF {card.components}"
                         + (f"  nodes {nodes[:10]}{' ...' if len(nodes) > 10 else ''}"))
        L.append(f"  constrained nodes (all SPC sets + GRID PS): {len(self.spc_nodes)}")
        for sid, cards in sorted(bdf.load_combinations.items()):
            L += [f"  {c.type} {sid}: " + " ".join(str(c).split()[2:]) for c in cards]
        for sid, cards in sorted(bdf.loads.items()):
            counts = Counter(c.type for c in cards)
            L.append(f"  load set {sid}: " + ", ".join(f"{k} {v}x" for k, v in counts.items()))
            for card in cards[:5]:
                L.append("      " + " ".join(card.raw_fields()[0:1] + [str(f) for f in card.raw_fields()[1:8]]))
        for sid, cards in sorted(bdf.dloads.items()):
            L += [f"  DLOAD {sid}: " + " ".join(str(f) for f in c.raw_fields()[2:]) for c in cards]
        for sid, cards in sorted(bdf.dload_entries.items()):
            L += [f"  {c.type} {sid}: " + " ".join(str(f) for f in c.raw_fields()[2:]) for c in cards]

        section("OTHER ANALYSIS CARDS")
        shown = set()
        for sid, cards in sorted(bdf.frequencies.items()):                # FREQ, FREQ1 ... FREQ5
            for card in cards:
                L += ["  " + line for line in str(card).splitlines() if line.strip() and not line.startswith("$")]
        for ctype in ("EIGRL", "EIGR", "TABDMP1", "TABLED1", "TABLED2", "TSTEP", "TSTEPNL", "NLPARM", "NLSTEP",
                      "PARAM", "MDLPRM", "DTI", "BCPARA"):
            if not 0 < bdf.card_count.get(ctype, 0) <= 20:
                continue
            try:
                cards = bdf.get_cards_by_card_types([ctype]).get(ctype, [])
            except Exception:
                continue
            for card in cards:
                text = "\n".join(line for line in str(card).splitlines() if line.strip() and not line.startswith("$"))
                L += ["  " + line for line in text.splitlines()]
                shown.add(ctype)
        for name, param in sorted(bdf.params.items()):
            if "PARAM" not in shown:
                L.append(f"  PARAM {name} = {param.values}")

        # ---- rigid elements, contact
        section("RIGID ELEMENTS, CONTACT")
        L.append("  rigid elements: " + (", ".join(f"{k} {v}x" for k, v in self.rigid_summary.items()) or "-"))
        dependent = Counter()
        for elem in bdf.rigid_elements.values():
            for n in elem.dependent_nodes:
                dependent[int(n)] += 1
        double = [n for n, c in dependent.items() if c > 1]
        if double:
            self.checks.append(("error", f"{len(double)} nodes are dependent in more than one rigid element: "
                                         f"{double[:10]}"))
        spc_dep = sorted(set(dependent) & set(self.spc_nodes))
        if spc_dep:
            self.checks.append(("error", f"{len(spc_dep)} dependent nodes of rigid elements are constrained by SPC: "
                                         f"{spc_dep[:10]}"))
        bsurf = getattr(bdf, "bsurf", {})
        L.append(f"  BSURF contact surfaces: {len(bsurf)}")
        for sid, card in sorted(bsurf.items()):
            sel = self.contact_body_ids == sid
            pids = sorted(set(self.property_ids[sel]))
            bodies = sorted(set(self.body_ids[sel]))
            L.append(f"      BSURF {sid:4d}: {len(card.eids):7d} elements, PID {pids}, mesh bodies {bodies}")
        contact_cards = {k: v for k, v in bdf.card_count.items() if k.startswith("BC") or k.startswith("BG")}
        if contact_cards:
            L.append("  contact cards: " + ", ".join(f"{k} {v}x" for k, v in sorted(contact_cards.items())))

        # ---- mesh checks
        section("MESH CHECKS")
        L += self._body_report()
        L += self._node_report()
        L += self._quality_report()

        # ---- unsupported cards
        rejected = getattr(bdf, "reject_count", {})
        if rejected:
            self.checks.append(("info", "cards not interpreted by pyNastran (kept, not checked): "
                                + ", ".join(f"{k} {v}x" for k, v in sorted(rejected.items()))))

        section("ALL CARDS")
        L += _table(["card", "count"], sorted(bdf.card_count.items()))

        # ---- check summary at the top
        order = {"error": 0, "warning": 1, "info": 2}
        summary = ["CHECK SUMMARY:"]
        for level, text in sorted(self.checks, key=lambda c: order.get(c[0], 3)):
            summary.append(f"  [{level.upper()}] {text}")
        if len(summary) == 1:
            summary.append("  no problems found")
        return "\n".join(summary + L) + "\n"

    @staticmethod
    def _comment_name(card):
        """Name of a material / property from the Patran comment above the card."""
        comment = getattr(card, "comment", "") or ""
        for line in comment.splitlines():
            for key in ("Material Record :", "region :"):
                if key in line:
                    return line.split(key, 1)[1].strip()
        return ""

    def _body_report(self):
        L = ["", "  Connected mesh bodies (elements sharing nodes):"]
        spc_idx = set(self.node_index(list(self.spc_nodes)).tolist()) if self.spc_nodes else set()
        spcd_idx = set(self.node_index([s[0] for s in self.spcd]).tolist()) if self.spcd else set()
        rigid_idx = set()
        for elem in self.bdf.rigid_elements.values():
            rigid_idx.update(self.node_index([int(n) for n in elem.independent_nodes + elem.dependent_nodes
                                              if n]).tolist())
        mass_idx = set(self.node_index([self._int(m[2].nid) for m in self.mass_points]).tolist())
        grid = self._grid
        offsets = np.asarray(grid.offset)
        conn = np.asarray(grid.cell_connectivity)
        cell_of_entry = np.repeat(np.arange(grid.n_cells), np.diff(offsets))
        rows = []
        for body in range(1, self.body_ids.max() + 1):
            sel = self.body_ids == body
            nodes = set(np.unique(conn[sel[cell_of_entry]]).tolist())
            links = []
            if nodes & spc_idx:
                links.append("SPC")
            if nodes & spcd_idx:
                links.append("SPCD")
            if nodes & rigid_idx:
                links.append("RBE")
            if nodes & mass_idx:
                links.append("CONM2")
            contact = sorted(set(self.contact_body_ids[sel]) - {0})
            if contact:
                links.append(f"BSURF {contact}")
            rows.append([body, int(np.count_nonzero(sel)), sorted(set(self.property_ids[sel].tolist())),
                         _fmt(np.nansum(self.volumes[sel])), _fmt(np.nansum(self.element_mass[sel])),
                         ", ".join(links) or "NOTHING"])
            if not ({"SPC", "SPCD", "RBE"} & set(links)) and not contact:
                self.checks.append(("warning", f"body {body} ({int(np.count_nonzero(sel))} elements) is not "
                                               f"connected by SPC / RBE / contact - free body?"))
        L += ["  " + line for line in _table(["body", "elements", "PID", "volume", "mass", "connected by"], rows)]
        if len(rows) > 1 and not getattr(self.bdf, "bsurf", {}):
            self.checks.append(("info", f"model has {len(rows)} separate mesh bodies and no contact definition"))
        return L

    def _node_report(self):
        L = [""]
        used = np.zeros(len(self.node_ids), dtype=bool)
        used[np.asarray(self._grid.cell_connectivity)] = True
        aux = set()
        for elem in self.bdf.rigid_elements.values():
            aux.update(int(n) for n in elem.independent_nodes + elem.dependent_nodes if n)
        aux.update(self._int(m[2].nid) for m in self.mass_points)
        aux.update(self.spc_nodes)
        aux_idx = self.node_index(sorted(aux)) if aux else np.array([], dtype=np.int64)
        aux_mask = np.zeros(len(self.node_ids), dtype=bool)
        aux_mask[aux_idx[aux_idx >= 0]] = True
        unused = ~used & ~aux_mask
        L.append(f"  nodes used by elements:              {np.count_nonzero(used)}")
        L.append(f"  nodes used only by RBE / CONM2 / SPC: {np.count_nonzero(~used & aux_mask)}")
        L.append(f"  unused nodes:                        {np.count_nonzero(unused)}"
                 + (f"  e.g. {self.node_ids[unused][:10].tolist()}" if unused.any() else ""))
        if unused.any():
            self.checks.append(("info", f"{np.count_nonzero(unused)} unused nodes"))

        # coincident nodes
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return L
        size = float(np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0)))
        tol = self.coincident_tolerance or 1e-6 * size
        idx = np.flatnonzero(used)
        pairs = np.array(sorted(cKDTree(self.points[idx]).query_pairs(tol)), dtype=np.int64).reshape(-1, 2)
        L.append(f"  coincident node pairs (distance < {_fmt(tol)}): {len(pairs)}")
        if len(pairs):
            pairs = idx[pairs]
            node_body = np.zeros(len(self.node_ids), dtype=np.int64)
            conn = np.asarray(self._grid.cell_connectivity)
            node_body[conn] = np.repeat(self.body_ids, np.diff(np.asarray(self._grid.offset)))
            same = node_body[pairs[:, 0]] == node_body[pairs[:, 1]]
            L.append(f"      between different bodies (contact / glue interfaces): {np.count_nonzero(~same)}")
            L.append(f"      inside one body (duplicate nodes?):                   {np.count_nonzero(same)}"
                     + (f"  e.g. {self.node_ids[pairs[same][:5]].tolist()}" if same.any() else ""))
            if same.any():
                self.checks.append(("warning", f"{np.count_nonzero(same)} coincident node pairs inside one body"))
        return L

    def _quality_report(self):
        L = ["", f"  Element quality (corner nodes; limits: scaled Jacobian < {self.MIN_SCALED_JACOBIAN}, "
                 f"edge ratio > {self.MAX_EDGE_RATIO}):"]
        rows = []
        type_names = np.array([f"{t} ({_vtk_name(c)})" for t, c in
                               zip(self.element_types, self._grid.celltypes)])
        for name in self.type_labels:
            sel = type_names == name
            sj, er = self.scaled_jacobian[sel], self.edge_ratio[sel]
            row = [name, int(np.count_nonzero(sel))]
            row += [_fmt(np.nanmin(sj), 3) if np.isfinite(sj).any() else "-",
                    int(np.count_nonzero(sj < self.MIN_SCALED_JACOBIAN)),
                    _fmt(np.nanmax(er), 3) if np.isfinite(er).any() else "-",
                    int(np.count_nonzero(er > self.MAX_EDGE_RATIO)),
                    int(np.count_nonzero(sj <= 0))]
            rows.append(row)
        L += ["  " + line for line in _table(["element", "count", "min SJ", f"SJ<{self.MIN_SCALED_JACOBIAN}",
                                               "max ER", f"ER>{self.MAX_EDGE_RATIO}", "inverted (SJ<=0)"], rows)]
        bad_sj = np.count_nonzero(self.scaled_jacobian < self.MIN_SCALED_JACOBIAN)
        inverted = np.count_nonzero(self.scaled_jacobian <= 0)
        if inverted:
            self.checks.append(("error", f"{inverted} inverted / degenerated elements (scaled Jacobian <= 0)"))
        if bad_sj:
            order = np.argsort(np.nan_to_num(self.scaled_jacobian, nan=9.0))[:min(bad_sj, 5)]
            worst = self.element_ids[order]
            self.checks.append(("warning", f"{bad_sj} elements with scaled Jacobian < {self.MIN_SCALED_JACOBIAN}, "
                                           f"worst: {worst.tolist()}"))
        bad_er = np.count_nonzero(self.edge_ratio > self.MAX_EDGE_RATIO)
        if bad_er:
            self.checks.append(("info", f"{bad_er} elements with edge ratio > {self.MAX_EDGE_RATIO}"))
        return L

    def save_report(self, file_path):
        Path(file_path).write_text(self.report, encoding="utf-8")
        self._log(f"Report saved to {file_path}")
