from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2_geom import OP2Geom
from pyNastran.op2.op2 import OP2
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

# PyTables / HDF5 is not thread-safe -> every access to an H5 file goes through this lock (the viewer reads results
# in background threads: loading of files, min / max of results, random response)
HDF5_LOCK = threading.RLock()

# h5 files - https://github.com/SteveDoyle2/pyNastran/issues/308

class ParseBdfFile:
    def __init__(self, file_path):

        try:
            # Initialize the BDF model
            self.model = BDF()

            # Read the BDF file
            self.model.read_bdf(file_path)

            print(f"File {file_path} was succesfully loaded")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def print_statistics(self):
        # Access information from the model
        print(f"Number of nodes: {len(self.model.nodes)}")
        print(f"Number of elements: {len(self.model.elements)}")
        print(f"Number of properties: {len(self.model.properties)}")
        print(f"Number of materials: {len(self.model.materials)}")
        print(f"Number of loads: {len(self.model.loads)}")

    def print_materials(self):
        print("File contains these materials:")
        for m in self.model.materials:
            print(self.model.materials[m])

    def print_loads(self):
        print("File contains these loads:")
        # for m in self.model.loads:
            # print(self.model.loads[m])
            # print(self.model.loads[m].__getattribute__())

        print("model")
        print(self.model)
        print("model.loads")
        print(self.model.loads)
        print(type(self.model.loads))
        print("other")
        for k in self.model.loads.keys():
            print(f"Key: {k}, load: {self.model.loads[k]}")


        # print(type(self.model.loads[6]))
        # print(type(self.model.loads[6][0]))
        #
        # print(type(self.model.loads[6][0].components))
        # print(self.model.loads[6][0].components)
        # print(self.model.loads[6][0].type)

        # print("list class print")
        # print(self.model.loads[1][0].N)
        # print(self.model.loads[1][0].type)
        #
        # self.model.loads[1][0].N[0] = 10.0
        #
        # print(self.model.loads[1][0].N)
        # for l in self.model.loads[1]:
        #     print(l.comment)
        #     a = l.comment.split(" ")
        #     print()
        #
        #     a[-1] = "kjsdkf\n"
        #
        #     tt = ""
        #     for o in a:
        #         tt = tt + o + " "
        #
        #     print(tt)
        #     print(type(l))
        #     print(l.scale)
        #     print(l.cid)
        #     print(l.sid)
        #     print(l.N)
        #     print(l.N[2])
        #     print(l.mb)
        #     print(l.cid_ref)
        #
        #     print(l.Cid())
        #     print(l.GravityVector())
        #     print(l.raw_fields())
        # # print(self.model.loads[0].sid)
        #
        # print("get")
        # print(self.model.get_reduced_loads())
        # print(self.model.get_)
        return list(self.model.loads.keys())

    def update_load_orientation(self, id, new: list, name=" "):
        print("Load to change:")
        print(self.model.loads[id])

        if self.model.loads[id][0].type == "GRAV":
            subname = "gravity"
            print(f"Old load orientation parameters: {self.model.loads[id][0].N}")

            if len(self.model.loads[id][0].N) != len(new):
                return f"Wrong number of new parameters, it has to be: {len(self.model.loads[id][0].N)}"

            for i in range(0,len(self.model.loads[id][0].N)):
                self.model.loads[id][0].N[i] = float(new[i])

            print(f"New load orientation parameters: {self.model.loads[id][0].N}")

        elif self.model.loads[id][0].type == "SPCD":
            subname = "acceleration"
            print(f"Old load orientation parameters: {self.model.loads[id][0].components}")

            non_zero_indices = [i+1 for i, val in enumerate(new) if val != 0]
            components = "".join(map(str, non_zero_indices))
            self.model.loads[id][0].components = [components]

            print(f"New load orientation parameters: {self.model.loads[id][0].components}")
        else:
            return "non defined load !!!"

        if name != " ":
            comment = self.model.loads[id][0].comment.split(" ")

            comment[-1] = subname + name + "\n"
            print(comment)
            comment.pop(0)
            new_text = " "
            for c in comment:
                print(c)
                new_text = new_text + c + " "

            self.model.loads[id][0].comment = new_text

        # self.model.loads[id][0].N[1] = float(y)
        # self.model.loads[id][0].N[2] = float(z)


        print(f"New comment: {self.model.loads[id][0].comment}")

    def update_material(self, ID, E = 0, rho=0, ny=0):
        print("Old material parameters:", end=" ")
        print(f"ID = {self.model.materials[ID].Mid()}, E = {self.model.materials[ID].e}, rho = {self.model.materials[ID].rho}, ny = {self.model.materials[ID].nu}")

        if E != 0:
            self.model.materials[1].e = float(E)
        if rho != 0:
            self.model.materials[1].rho = float(rho)
        if ny != 0:
            self.model.materials[1].ny = float(ny)

        print("Mew material parameters:", end=" ")
        print(
            f"ID = {self.model.materials[ID].Mid()}, E = {self.model.materials[ID].e}, rho = {self.model.materials[ID].rho}, ny = {self.model.materials[ID].nu}")

    def save_bdf_file(self, file_path):
        self.model.write_bdf(file_path)
        print(f"File {file_path} was saved")


class ParseOP2File:
    def __init__(self, file_name, h5=False):
        """
            Extracts accelerations from an OP2 file.

            Args:
                op2_file (str): Path to the OP2 file.

            Returns:
                dict: Parsed acceleration data.
            """
        op2 = OP2()

        try:
            # Read the OP2 file
            # if h5:
            #     op2.load_hdf5_file(file_name)
            # else:
            #     op2.read_op2(file_name)
            op2.read_op2(file_name)

            # Extract acceleration data
            self.acceleration_data = op2.accelerations

            if not self.acceleration_data:
                print("No acceleration data found in the OP2 file.")

            subcase = 1
            # Print acceleration data for each subcase
            # for subcase, accelerations in self.acceleration_data.items():
            accelerations = self.acceleration_data[1]

            self.df = accelerations.freqs
            self.complex = accelerations.data
            self.real_part = accelerations.data.real
            self.imag_part = accelerations.data.imag

            print(f"File {file_name} was succesfully loaded")
            print(f"Subcase {subcase} loaded")
            print(f"File contains {len(self.complex[0])} nodes")

        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_magnitude_acceleration(self, node):
        """
        Calculates the resultant acceleration magnitude for a given node.

        Args:
            node (int): The node number for which to calculate the resultant acceleration.

        Returns:
            list[float]: A list of resultant acceleration magnitudes at each frequency step.
        """

        accX = self.get_acc_node_complex_data(node, "X")
        accY = self.get_acc_node_complex_data(node, "Y")
        accZ = self.get_acc_node_complex_data(node, "Z")


        acc = [abs(complex(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)) ** 0.5 for acc_x, acc_y, acc_z in zip(accX, accY, accZ)]

        # acc = []
        # for i in range(0,len(accX)):
        #     aa = cmath.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        #     acc.append(cmath.polar(aa)[0])

        return acc

    def get_acc_node_data(self, node, axis):
        """
            Calculates the magnitudes of complex acceleration data for a specific axis.

        Args:
            node (int): The node number.
            axis (str): The axis ('X', 'Y', 'Z') to get acceleration data for.

        Returns:
            list[float]: A list of magnitudes for the complex acceleration data.
        """

        acc_axis = self.get_acc_node_complex_data(node, axis)
        return [abs(value) for value in acc_axis]

    def get_acc_node_complex_data(self, node, axis):
        """
        Extracts acceleration data for a specific node and axis from a 3D dataset.

        Args:
            node (int): Node index (1-based).
            axis (str): Axis to extract ("X", "Y", or "Z").

        Returns:
            List[float]: List of acceleration values for the specified node and axis.

        Raises:
            ValueError: If the axis is invalid or the node index is out of bounds.
        """

        # Map axis to index
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Choose from 'X', 'Y', or 'Z'.")

        axis_par = axis_map[axis]

        node_indices = self.acceleration_data[1].node_gridtype[:, 0]
        node_index = -1
        try:
            if node in node_indices:
                node_index = list(node_indices).index(node)
        except:
            raise ValueError(f"Node index {node} is out of bounds for the dataset.")

        # print(f"File contain {len(self.complex[0])} nodes")
        # print(f"Node {node} has index {node_index}")

        return [row[node_index][axis_par] for row in self.complex]

    def get_frequencies(self):
        return self.df


def read_pch_file(pch_file):
    """
    Reads a PCH file and extracts its content.

    Args:
        pch_file (str): Path to the PCH file.
    """
    # Load the OP2Geom parser (PCH is similar in structure to OP2 data)
    model = OP2Geom()

    # Read the PCH file
    model.read_op2(pch_file)

    # Access results (e.g., displacements, forces, etc.)
    for key, value in model.displacements.items():
        print(f"Displacement for key {key}: {value}")

    for key, value in model.stresses.items():
        print(f"Stress for key {key}: {value}")

    for key, value in model.forces.items():
        print(f"Force for key {key}: {value}")


# =====================================================================================================================
#   Nastran mesh -> pyvista (NastranMesh) and MSC Nastran HDF5 (*.h5) results (ParseH5File)
# =====================================================================================================================
#   Notes:
#   * Datasets in Nastran H5 files are compressed by the BLOSC filter (HDF5 filter id 32001). PyTables ("tables")
#     contains BLOSC, plain h5py can list the file structure but cannot read the data without "hdf5plugin"
#     -> all data below are read by PyTables.
#   * Element connectivity tables have a fixed width (CTETRA 10, CPENTA 15, CHEXA 20 nodes), unused (midside) nodes
#     are stored as 0 -> the number of nodes (linear / quadratic element) is detected for every element.
#   * Node IDs are not continuous -> results are always mapped by node / element ID, never by row index.
#   * All elements are put into ONE pyvista UnstructuredGrid sharing one point array (no merging of grids),
#     so the order of points = order of GRIDs and nodal results can be assigned directly.


class NastranMesh:
    """
    Mesh (GRID + elements) of a Nastran model converted into ONE pyvista UnstructuredGrid (all element types share
    one point array, order of points = order of GRIDs, order of cells = element_ids) + common results interface.
    Base class of ParseH5File (results from *.h5) and Preprocess.bdf_model.BdfModel (input *.bdf).

    Interface used by the viewer (visualization/h5_viewer.py):
        file_path, node_ids, points, element_ids, property_ids, element_types, to_pyvista(linear),
        results [(name, "point" / "cell")], result_meta {name: (group, item)}, steps [{"label", "kind", ...}],
        values(name, step, mode), history(name, index, step, mode), is_complex(name), step_axis(step),
        get_deformation(step, phase), deformation_vectors(step), modes, categorical_results, result_annotations,
        results_title, add_static_result(...)
    """

    _ELEMENT_TYPES = {
        "CTETRA": (4, 10, 10, 24, None),
        "CPYRAM": (5, 14, 13, 27, None),
        "CPENTA": (6, 13, 15, 26, [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11]),
        "CHEXA": (8, 12, 20, 25, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15]),
        "CTRIA3": (3, 5, 3, 5, None),
        "CTRIAR": (3, 5, 3, 5, None),
        "CTRIA6": (3, 5, 6, 22, None),
        "CQUAD4": (4, 9, 4, 9, None),
        "CQUADR": (4, 9, 4, 9, None),
        "CQUAD8": (4, 9, 8, 23, None),
        "CROD": (2, 3, 2, 3, None),
        "CONROD": (2, 3, 2, 3, None),
    }
    _VTK_NAMES = {3: "LINE2", 5: "TRIA3", 9: "QUAD4", 10: "TETRA4", 12: "HEXA8", 13: "PENTA6", 14: "PYRAM5",
                  22: "TRIA6", 23: "QUAD8", 24: "TETRA10", 25: "HEXA20", 26: "PENTA15", 27: "PYRAM13"}
    _NODAL_LABELS = {"DISPLACEMENT": "Displacement", "EIGENVECTOR": "Eigenvector", "VELOCITY": "Velocity",
                     "ACCELERATION": "Acceleration", "SPC_FORCE": "SPC force", "MPC_FORCE": "MPC force",
                     "APPLIED_LOAD": "Applied load"}
    _SOLID_COMPONENTS = ("X", "Y", "Z", "TXY", "TYZ", "TZX")
    _QUADRATIC_CELLS = {
        22: (5, 3, [(3, 0, 1), (4, 1, 2), (5, 2, 0)]),
        23: (9, 4, [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)]),
        24: (10, 4, [(4, 0, 1), (5, 1, 2), (6, 2, 0), (7, 0, 3), (8, 1, 3), (9, 2, 3)]),
        25: (12, 8, [(8, 0, 1), (9, 1, 2), (10, 2, 3), (11, 3, 0), (12, 4, 5), (13, 5, 6), (14, 6, 7), (15, 7, 4),
                     (16, 0, 4), (17, 1, 5), (18, 2, 6), (19, 3, 7)]),
        26: (13, 6, [(6, 0, 1), (7, 1, 2), (8, 2, 0), (9, 3, 4), (10, 4, 5), (11, 5, 3), (12, 0, 3), (13, 1, 4),
                     (14, 2, 5)]),
        27: (14, 5, [(5, 0, 1), (6, 1, 2), (7, 2, 3), (8, 3, 0), (9, 0, 4), (10, 1, 4), (11, 2, 4), (12, 3, 4)]),
    }

    def __init__(self, file_path, verbose=True):
        self.file_path = Path(file_path)
        self.verbose = verbose
        if not self.file_path.is_file():
            raise FileNotFoundError(f"File not found: {self.file_path.resolve()}")

        self.messages = []          # info about skipped / unsupported data
        self.results = []           # [(result name, "point" / "cell")] - quantities shown in the viewer
        self.result_meta = {}       # result name -> (group, item), e.g. ("Stress (element)", "von Mises")
        self.categorical_results = {"Property ID"}  # results shown with discrete colors
        self.result_annotations = {}                # result name -> {value: label} for the color bar
        self.results_title = "results"
        # result steps (time steps, frequencies, modes, subcases): [{"domain", "subcase", "kind", "x", "label"}]
        # empty list = only one state (input model, static fields)
        self.steps = []
        self.domains = {}           # result domain ID -> label
        # natural frequencies (eigenvalue summary): {"mode", "freq", "mass", "stiff"} -> arrays, {} = not available
        self.modes = {}
        self._static_values = {}    # results added by add_static_result (e.g. random RMS maps)
        # numpy copies of the result arrays stored in the grid - values() reads them without VTK (VTK objects must
        # not be used from background threads while the 3D view renders - possible deadlock with newer VTK)
        self._grid_values = {}
        self._grid = None
        self._grid_linear = None
        self._cell_blocks = []      # [(vtk type, point indices (n, k))]
        self._element_parts = []    # [(element IDs, property IDs, element type names)] - see _add_elements
        self.element_counts = {}    # "CTETRA (TETRA10)" -> count

    # ---------------------------------------------------------------------------------------------- mesh
    def _log(self, text):
        if self.verbose:
            print(text)

    @staticmethod
    def _lookup(sorted_ids, order, ids):
        """Returns (indices, found mask) of IDs in an array given by its sorted IDs and argsort order."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(sorted_ids) == 0:
            return np.zeros(ids.shape, dtype=np.int64), np.zeros(ids.shape, dtype=bool)
        pos = np.clip(np.searchsorted(sorted_ids, ids), 0, len(sorted_ids) - 1)
        return order[pos], sorted_ids[pos] == ids

    def node_index(self, node_ids):
        """Indices of nodes (= pyvista point indices) for given node IDs, -1 for unknown IDs."""
        idx, found = self._lookup(self._node_sorted, self._node_order, node_ids)
        return np.where(found, idx, -1)

    def cell_index(self, element_ids):
        """Indices of elements (= pyvista cell indices) for given element IDs, -1 for unknown IDs."""
        idx, found = self._lookup(self._cell_sorted, self._cell_order, element_ids)
        return np.where(found, idx, -1)

    def _set_nodes(self, node_ids, points):
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.points = np.asarray(points, dtype=np.float64)
        self._node_order = np.argsort(self.node_ids)
        self._node_sorted = self.node_ids[self._node_order]

    def _element_blocks(self, name, G):
        """Splits element table into blocks of linear / quadratic elements -> [(vtk type, node IDs, row mask)]"""
        n_corner, vtk_lin, n_quad, vtk_quad, perm = self._ELEMENT_TYPES[name]
        G = G.reshape(len(G), -1)
        n_nodes = np.count_nonzero(G, axis=1)

        is_quad = np.zeros(len(G), dtype=bool)
        if n_quad > n_corner and G.shape[1] >= n_quad:
            is_quad = n_nodes == n_quad
        partial = ~is_quad & (n_nodes != n_corner)
        if partial.any():
            self.messages.append(f"{name}: {np.count_nonzero(partial)} elements with incomplete midside nodes "
                                 f"are shown as linear")

        blocks = []
        if is_quad.any():
            conn = G[is_quad, :n_quad]
            if perm is not None:
                conn = conn[:, perm]
            blocks.append((vtk_quad, conn, is_quad))
        if (~is_quad).any():
            blocks.append((vtk_lin, G[~is_quad, :n_corner], ~is_quad))
        return blocks

    def _add_elements(self, name, eids, pids, nodes):
        """
        Adds elements of one Nastran type.

        Args:
            name (str): Nastran element name (CTETRA, CHEXA, ...); 2-node elements (CBAR, CBUSH, ...) are lines
            eids, pids: element and property IDs (n,)
            nodes: node IDs (n, k), unused (midside) nodes = 0
        """
        eids = np.asarray(eids, dtype=np.int64)
        pids = np.asarray(pids, dtype=np.int64)
        nodes = np.asarray(nodes, dtype=np.int64).reshape(len(eids), -1)
        if name in self._ELEMENT_TYPES:
            blocks = self._element_blocks(name, nodes)
        elif nodes.shape[1] == 2:
            blocks = [(3, nodes, np.ones(len(eids), dtype=bool))]
        else:
            self.messages.append(f"{name} ({len(eids)}x): not shown (unsupported element)")
            return

        for vtk_type, conn_ids, mask in blocks:
            idx, found = self._lookup(self._node_sorted, self._node_order, conn_ids)
            ok = found.all(axis=1)
            if not ok.all():
                self.messages.append(f"{name}: {np.count_nonzero(~ok)} elements reference unknown nodes (skipped)")
            self._cell_blocks.append((vtk_type, idx[ok]))
            self._element_parts.append((eids[mask][ok], pids[mask][ok], np.full(np.count_nonzero(ok), name)))
            key = f"{name} ({self._VTK_NAMES.get(vtk_type, vtk_type)})"
            self.element_counts[key] = self.element_counts.get(key, 0) + int(np.count_nonzero(ok))

    def _finish_elements(self):
        if not self._cell_blocks:
            raise ValueError(f"No supported elements found in {self.file_path.name}")
        self.element_ids = np.concatenate([p[0] for p in self._element_parts])
        self.property_ids = np.concatenate([p[1] for p in self._element_parts])
        self.element_types = np.concatenate([p[2] for p in self._element_parts])
        self._cell_order = np.argsort(self.element_ids)
        self._cell_sorted = self.element_ids[self._cell_order]

    def _fill_midside_nodes(self, node_values):
        """Element results are given in corner nodes only -> midside nodes of quadratic elements get the average
        of the two corner nodes of their edge (otherwise they stay NaN and spoil sections / interpolation)."""
        missing = ~np.isfinite(node_values).all(axis=1)
        for vtk_type, conn in self._cell_blocks:
            for mid, a, b in self._QUADRATIC_CELLS.get(vtk_type, (None, None, []))[2]:
                sel = missing[conn[:, mid]]
                node_values[conn[sel, mid]] = 0.5 * (node_values[conn[sel, a]] + node_values[conn[sel, b]])

    # ---------------------------------------------------------------------------------------------- derived values
    @staticmethod
    def von_mises(s):
        """von Mises stress from array (n, 6) of components X, Y, Z, TXY, TYZ, TZX."""
        sx, sy, sz, txy, tyz, tzx = s.T
        return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))

    @staticmethod
    def principal(s):
        """Principal values (n, 3) sorted min -> max from array (n, 6) of components X, Y, Z, TXY, TYZ, TZX."""
        out = np.full((len(s), 3), np.nan)
        ok = np.isfinite(s).all(axis=1)
        sx, sy, sz, txy, tyz, tzx = s[ok].T
        tensor = np.stack([np.stack([sx, txy, tzx], -1), np.stack([txy, sy, tyz], -1), np.stack([tzx, tyz, sz], -1)], 1)
        out[ok] = np.linalg.eigvalsh(tensor)
        return out

    @staticmethod
    def complex_part(values, mode):
        """Complex values -> "amplitude", "phase" [deg], "real", "imag"; real values are returned unchanged."""
        values = np.asarray(values)
        if not np.iscomplexobj(values):
            return values.astype(float)
        if mode == "phase":
            return np.degrees(np.angle(values))
        if mode == "real":
            return values.real
        if mode == "imag":
            return values.imag
        return np.abs(values)

    @staticmethod
    def vector_magnitude(vectors, mode):
        """Magnitude of vectors (n, k); complex vectors: amplitude = sqrt(sum |c|^2), real / imag = norm of the
        real / imaginary parts, phase = not defined (NaN)."""
        vectors = np.asarray(vectors)
        if not np.iscomplexobj(vectors):
            return np.linalg.norm(vectors, axis=1)
        if mode == "phase":
            return np.full(len(vectors), np.nan)
        if mode == "real":
            return np.linalg.norm(vectors.real, axis=1)
        if mode == "imag":
            return np.linalg.norm(vectors.imag, axis=1)
        return np.sqrt(np.sum(np.abs(vectors) ** 2, axis=1))

    # ---------------------------------------------------------------------------------------------- results API
    def _register(self, name, location, group, item):
        self.results.append((name, location))
        self.result_meta[name] = (group, item)

    def result_location(self, name):
        for result, location in self.results:
            if result == name:
                return location
        raise KeyError(name)

    def is_complex(self, name):
        """True for results with complex values (frequency response)."""
        return False

    def add_static_result(self, name, location, group, item, values):
        """
        Adds (or replaces) a result which does not depend on the result step (e.g. RMS of a random response).

        Args:
            name (str): result name, location: "point" (n_nodes values) / "cell" (n_elements values)
            group, item (str): group and label in the table of results
        """
        values = np.asarray(values, dtype=float)
        if name not in self.result_meta:
            self._register(name, location, group, item)
        else:
            self.result_meta[name] = (group, item)
        self._static_values[name] = values                     # (save_vtu saves them too)

    def values(self, name, step=0, mode="amplitude"):
        """Values of the result `name` in result step `step` (array n_nodes for "point" / n_elements for "cell"
        results). Base class: static arrays stored in the grid (Property ID, model fields of a BDF ...)."""
        if name in self._static_values:
            return self._static_values[name]
        if name in self._grid_values:
            return self._grid_values[name]
        grid = self.to_pyvista()
        data = grid.point_data if self.result_location(name) == "point" else grid.cell_data
        return np.asarray(data[name], dtype=float)

    def history(self, name, index, step=0, mode="amplitude"):
        """
        Course of the result `name` in node / element `index` (point or cell index) over the steps of the same
        subcase as `step` (time, frequency, modes) -> (step indices, x values, y values) or None.
        """
        return None

    def step_axis(self, step=0):
        """Label of the x axis for history plots of the steps of the subcase of `step`."""
        return ""

    def get_deformation(self, step=0, phase=0.0):
        """Translational displacement (n_nodes, 3) in result step `step` (complex results: real part of
        U * exp(i * phase)), None if not available."""
        return None

    def deformation_vectors(self, step=0):
        """Translational displacement (n_nodes, 3) in result step `step` - complex for frequency response (used
        for the animation: real part of U * exp(i * phase)), None if not available."""
        return self.get_deformation(step)

    # ---------------------------------------------------------------------------------------------- output
    @staticmethod
    def _make_grid(blocks, points):
        import pyvista as pv

        cells = np.concatenate([np.column_stack([np.full(len(c), c.shape[1]), c]).ravel() for _, c in blocks])
        cell_types = np.concatenate([np.full(len(c), t, dtype=np.uint8) for t, c in blocks])
        return pv.UnstructuredGrid(cells, cell_types, points)

    def _linear_blocks(self):
        """Cell blocks with quadratic elements converted to linear ones (corner nodes only)."""
        blocks = []
        for vtk_type, conn in self._cell_blocks:
            if vtk_type in self._QUADRATIC_CELLS:
                vtk_type, n_corner, _ = self._QUADRATIC_CELLS[vtk_type]
                conn = conn[:, :n_corner]
            blocks.append((vtk_type, conn))
        return blocks

    def to_pyvista(self, linear=False):
        """
        Returns pyvista.UnstructuredGrid of the mesh with node / element / property IDs (and static fields of
        subclasses). Results of a step are added by save_vtu() or can be got by values().

        Args:
            linear (bool): True -> quadratic elements are converted to linear ones (corner nodes only, same order
                of points and cells, same data arrays) - much faster sections / clipping, used by the viewer.
        """
        if linear:
            if self._grid_linear is None:
                grid = self.to_pyvista()
                self._grid_linear = self._make_grid(self._linear_blocks(), self.points)
                self._grid_linear.GetPointData().ShallowCopy(grid.GetPointData())   # shares the data arrays
                self._grid_linear.GetCellData().ShallowCopy(grid.GetCellData())
            return self._grid_linear

        if self._grid is not None:
            return self._grid

        grid = self._make_grid(self._cell_blocks, self.points)
        grid.point_data["Node ID"] = self.node_ids
        grid.cell_data["Element ID"] = self.element_ids
        grid.cell_data["Property ID"] = self.property_ids
        self._grid_values["Property ID"] = self.property_ids.astype(float)
        if "Property ID" not in self.result_meta:
            self._register("Property ID", "cell", "Model", "Property ID")
        self._grid = grid
        return grid

    def save_vtu(self, file_path, step=0, mode="amplitude"):
        """Saves mesh + all results of result step `step` to VTK XML file (*.vtu) - can be opened e.g. in ParaView."""
        grid = self.to_pyvista().copy()
        for name, location in self.results:
            data = grid.point_data if location == "point" else grid.cell_data
            data[name] = self.values(name, step, mode)
        grid.save(str(file_path))
        self._log(f"File {file_path} was saved")

    def print_summary(self):
        print(f"File: {self.file_path}")
        print(f"  nodes: {len(self.node_ids)}")
        print(f"  elements: {len(self.element_ids)}")
        for key, count in self.element_counts.items():
            print(f"      {key}: {count}")
        if self.steps:
            kinds = sorted({s["kind"] for s in self.steps})
            print(f"  result steps: {len(self.steps)} ({', '.join(kinds)}), {self.steps[0]['label']} ... "
                  f"{self.steps[-1]['label']}")
        print(f"  results: {len(self.results)}")
        for m in self.messages:
            print(f"  ! {m}")


class ParseH5File(NastranMesh):
    """
    Reads mesh (GRID + elements) and results from an MSC Nastran HDF5 file (MDLPRM,HDF5,1 in bulk data).

    Results are read on demand (the file stays open, close() closes it), so also big transient / frequency
    response results can be browsed:
        * result steps: self.steps - static subcases, load / time steps, frequencies, modes
        * nodal results: every table in /NASTRAN/RESULT/NODAL with components X..RZ (DISPLACEMENT, VELOCITY,
          ACCELERATION, SPC_FORCE, EIGENVECTOR, ...), real or complex (*_CPLX - frequency response)
        * element results: STRESS / STRAIN of solid elements (TETRA, PENTA, HEXA, PYRAM), real or complex
        * values(name, step, mode) - values of a result in one step (complex: amplitude / phase / real / imag)
        * history(name, index, step, mode) - course of a result in one node / element over time / frequency

    Supported elements: CTETRA, CPYRAM, CPENTA, CHEXA (linear and quadratic), CTRIA3/6, CQUAD4/8, CROD, CONROD,
    line elements with GA/GB (CBAR, CBEAM, CBUSH, ...); others (CONM2, RBE2, ...) are not shown.

    Example:
        h5 = ParseH5File("box2_weight1g_z_ch.h5")
        h5.print_summary()
        u = h5.values("Displacement magnitude", step=0)       # array (n_nodes,)
        steps, x, y = h5.history("Acceleration Z", h5.node_index([1234])[0])   # frequency response of node 1234
    """

    _NODAL_COMPONENTS = ("X", "Y", "Z", "RX", "RY", "RZ")
    _CACHE_BYTES = 800e6                                        # memory for cached result arrays

    def __init__(self, file_path, verbose=True, progress=None):
        """
        Args:
            file_path: MSC Nastran HDF5 result file
            verbose (bool): print messages
            progress: optional function progress(fraction 0..1, text) called during loading (background loading
                in the viewer; it may raise an exception to cancel the loading)
        """
        import tables  # PyTables - reads the BLOSC compressed Nastran datasets

        super().__init__(file_path, verbose)
        report = progress or (lambda fraction, text: None)
        self._tables = {}               # result table key -> description (path, blocks of domains, ...)
        self._quantities = {}           # result name -> definition (source table, components, operation)
        self._cache = OrderedDict()     # LRU cache of result arrays
        self._cache_lock = threading.RLock()
        self._row_maps = {}             # (table key, domain) -> (sorted IDs, order) for single rows (history)
        self._adjacent = None           # node -> cells lookup (nodal averaged histories)
        self._h5 = None
        self.reopen = True              # False = the file is not opened again after close() (viewer closed it)
        report(0.0, "opening the file")
        with HDF5_LOCK:
            self._h5 = tables.open_file(str(self.file_path), mode="r")
        try:
            h5 = self._h5
            report(0.02, "reading nodes")
            with HDF5_LOCK:
                self._read_nodes(h5)
            self._read_elements(h5, lambda k, n, name: report(0.05 + 0.45 * k / max(n, 1), f"reading {name}"))
            report(0.5, "reading result domains")
            with HDF5_LOCK:
                self._read_domains(h5)
                self._scan_results(h5)
            self._make_steps()
            report(0.6, "checking result tables")
            self._register_results()
            report(0.75, "building the mesh")
            self.to_pyvista()                                   # mesh + Property ID
        except BaseException:
            self.close()                                        # do not keep the file locked after an error
            raise
        self._log(f"File {self.file_path.name} was succesfully loaded")

    def _file(self):
        """Open H5 file (reopened after close())."""
        import tables

        with HDF5_LOCK:
            if self._h5 is None or not self._h5.isopen:
                if not self.reopen:
                    raise RuntimeError(f"{self.file_path.name} was closed")
                self._h5 = tables.open_file(str(self.file_path), mode="r")
            return self._h5

    def close(self):
        """Closes the H5 file (it is opened again automatically when other results are needed)."""
        with HDF5_LOCK:
            if self._h5 is not None and self._h5.isopen:
                self._h5.close()

    # ---------------------------------------------------------------------------------------------- mesh
    def _read_nodes(self, h5):
        grid = h5.get_node("/NASTRAN/INPUT/NODE/GRID").read()
        self._set_nodes(grid["ID"], grid["X"])

        n_cp = np.count_nonzero(grid["CP"])
        n_cd = np.count_nonzero(grid["CD"])
        if n_cp:
            self.messages.append(f"{n_cp} GRIDs have CP != 0 - coordinates are NOT transformed to the basic system!")
        if n_cd:
            self.messages.append(f"{n_cd} GRIDs have CD != 0 - their nodal results are in local coordinate systems!")

    def _read_elements(self, h5, progress=None):
        import tables

        with HDF5_LOCK:
            nodes = list(h5.list_nodes("/NASTRAN/INPUT/ELEMENT"))
        for k, tbl in enumerate(nodes):
            name = tbl._v_name
            if progress is not None:
                progress(k, len(nodes), name)
            if not isinstance(tbl, tables.Table):
                self.messages.append(f"{name}: not shown (unsupported element)")
                continue
            with HDF5_LOCK:
                data = tbl.read()
            fields = data.dtype.names
            pids = data["PID"] if "PID" in fields else np.zeros(len(data), dtype=np.int64)
            if name in self._ELEMENT_TYPES and "G" in fields:
                self._add_elements(name, data["EID"], pids, data["G"])
            elif "GA" in fields and "GB" in fields:
                self._add_elements(name, data["EID"], pids, np.column_stack([data["GA"], data["GB"]]))
            else:
                self.messages.append(f"{name} ({len(data)}x): not shown (unsupported element)")
        self._finish_elements()

    # ---------------------------------------------------------------------------------------------- steps
    def _read_domains(self, h5):
        self._domain_info = {}
        path = "/NASTRAN/RESULT/DOMAINS"
        if path in h5:
            dom = h5.get_node(path).read()
            fields = dom.dtype.names
            for row in dom:
                def get(field):
                    return row[field] if field in fields else 0

                self._domain_info[int(row["ID"])] = dict(subcase=int(get("SUBCASE")), analysis=int(get("ANALYSIS")),
                                                         x=float(get("TIME_FREQ_EIGR")), mode=int(get("MODE")))
        self._mode_freq = {}                                    # mode number -> natural frequency
        path = "/NASTRAN/RESULT/SUMMARY/EIGENVALUE"
        if path in h5:
            ev = h5.get_node(path).read()
            fields = ev.dtype.names
            if "MODE" in fields and "FREQ" in fields:
                self._mode_freq = {int(m): float(f) for m, f in zip(ev["MODE"], ev["FREQ"])}
                order = np.argsort(ev["MODE"], kind="stable")
                self.modes = {"mode": np.asarray(ev["MODE"][order], dtype=int),
                              "freq": np.asarray(ev["FREQ"][order], dtype=float)}
                for field, key in (("MASS", "mass"), ("STIFF", "stiff")):
                    if field in fields:
                        self.modes[key] = np.asarray(ev[field][order], dtype=float)

    def enforced_motion(self):
        """Enforced motion of the model (SPCD cards) -> list of (node ID, components e.g. "2" / "123", value)."""
        path = "/NASTRAN/INPUT/CONSTRAINT/SPCD"
        with HDF5_LOCK:
            h5 = self._file()
            if path not in h5:
                return []
            data = h5.get_node(path).read()
        fields = data.dtype.names
        if not all(f in fields for f in ("G", "C", "D")):
            return []
        return [(int(g), str(int(c)), float(d)) for g, c, d in zip(np.ravel(data["G"]), np.ravel(data["C"]),
                                                                    np.ravel(data["D"]))]

    @staticmethod
    def _domain_blocks(h5, tbl):
        """Rows of every result domain in a table: {domain: (start, stop)} (or array of rows if not contiguous)."""
        index_path = "/INDEX" + tbl._v_pathname
        if index_path in h5:
            idx = h5.get_node(index_path).read()
            return {int(d): (int(p), int(p) + int(n)) for d, p, n in zip(idx["DOMAIN_ID"], idx["POSITION"],
                                                                        idx["LENGTH"])}
        if "DOMAIN_ID" not in tbl.dtype.names:
            return {0: (0, tbl.nrows)}
        dom = tbl.col("DOMAIN_ID")
        change = np.flatnonzero(np.diff(dom)) + 1
        blocks = {}
        for start, stop in zip(np.concatenate([[0], change]), np.concatenate([change, [len(dom)]])):
            d = int(dom[start])
            if d in blocks:                                     # domain split into more parts
                old = blocks[d]
                old = np.arange(*old) if isinstance(old, tuple) else old
                blocks[d] = np.concatenate([old, np.arange(start, stop)])
            else:
                blocks[d] = (int(start), int(stop))
        return blocks

    def _scan_results(self, h5):
        """Finds supported result tables (nothing is read except of the domain positions)."""
        import tables

        path = "/NASTRAN/RESULT/NODAL"
        if path in h5:
            for tbl in h5.list_nodes(path):
                if not isinstance(tbl, tables.Table):
                    continue
                name = tbl._v_name
                fields = tbl.dtype.names
                real = [c for c in self._NODAL_COMPONENTS if c in fields]
                cplx = [c for c in self._NODAL_COMPONENTS if c + "R" in fields and c + "I" in fields]
                if "ID" not in fields or not (real or cplx):
                    if name != "GRID_WEIGHT":
                        self.messages.append(f"Nodal result {name}: unsupported format - skipped")
                    continue
                self._tables[name] = dict(path=tbl._v_pathname, kind="nodal", complex=bool(cplx),
                                          base=name[:-5] if name.endswith("_CPLX") else name, comps=cplx or real,
                                          id_field="ID", blocks=self._domain_blocks(h5, tbl))
        else:
            self.messages.append("No nodal results in the file")

        for base in ("STRESS", "STRAIN"):
            for suffix in ("", "_CPLX"):
                path = f"/NASTRAN/RESULT/ELEMENTAL/{base}{suffix}"
                if path not in h5:
                    continue
                for tbl in h5.list_nodes(path):
                    if not isinstance(tbl, tables.Table):
                        continue
                    fields = tbl.dtype.names
                    real = all(c in fields for c in self._SOLID_COMPONENTS)
                    cplx = all(c + "R" in fields and c + "I" in fields for c in self._SOLID_COMPONENTS)
                    if not ("EID" in fields and "GRID" in fields and (real or cplx)):
                        self.messages.append(f"{base}{suffix} {tbl._v_name}: not supported (only solid elements)")
                        continue
                    self._tables[f"{base}{suffix}/{tbl._v_name}"] = dict(
                        path=tbl._v_pathname, kind="element", complex=cplx, base=base, comps=self._SOLID_COMPONENTS,
                        id_field="EID", element="C" + tbl._v_name, blocks=self._domain_blocks(h5, tbl))

    @staticmethod
    def _table_family(name):
        """Results of a Nastran RANDOM analysis written to H5 (tables *_PSD, *_CRM, *_RMS, *_NO) -> "psd" / "rms",
        other tables -> ""."""
        tokens = set(name.upper().replace("/", "_").split("_"))
        if tokens & {"RMS", "NO"}:
            return "rms"
        if tokens & {"PSD", "CRM"}:
            return "psd"
        return ""

    def _nodal_label(self, base):
        """DISPLACEMENT -> Displacement, ACCELERATION_PSD -> Acceleration PSD, ..."""
        words = base.split("_")
        for k in range(len(words), 0, -1):
            head = "_".join(words[:k])
            if head in self._NODAL_LABELS:
                return " ".join([self._NODAL_LABELS[head]] + words[k:])
        return base.replace("_", " ").capitalize()

    def _make_steps(self):
        domains = sorted({d for info in self._tables.values() for d in info["blocks"]})
        complex_domains = {d for info in self._tables.values() if info["complex"] for d in info["blocks"]}
        eigen_domains = {d for info in self._tables.values() if info["base"] == "EIGENVECTOR" for d in info["blocks"]}
        families = defaultdict(set)                             # domain -> families of the tables containing it
        for key, info in self._tables.items():
            for d in info["blocks"]:
                families[d].add(self._table_family(key))
        steps = []
        for d in domains:
            info = self._domain_info.get(d, dict(subcase=0, analysis=0, x=0.0, mode=0))
            family = next(iter(families[d])) if len(families[d]) == 1 else ""
            steps.append(dict(domain=d, subcase=info["subcase"], x=info["x"], mode=info["mode"], family=family,
                              analysis=info["analysis"], complex=d in complex_domains, eigen=d in eigen_domains))
        subcases = defaultdict(list)
        for s in steps:
            subcases[(s["subcase"], s["family"])].append(s)
        for (sc, family), group in subcases.items():
            if family == "rms":
                kind = "static"
            elif family == "psd":
                kind = "frequency"
            elif any(s["analysis"] == 5 or s["complex"] for s in group):
                kind = "frequency"
            elif any(s["analysis"] == 2 or s["eigen"] for s in group) and any(s["mode"] for s in group):
                kind = "mode"
            elif len(group) > 1:
                kind = "time"
            else:
                kind = "static"
            prefix = {"rms": "RANDOM RMS", "psd": "RANDOM PSD"}.get(family)
            for s in group:
                s["kind"] = kind
                s["series"] = (sc, kind, family)                # steps of one course (time, frequency, modes)
                if prefix and kind == "static":
                    s["label"] = f"SC {sc}, {prefix}"
                elif prefix:
                    s["label"] = f"SC {sc}, {prefix} f = {s['x']:.5g} Hz"
                elif kind == "mode":
                    s["x"] = float(s["mode"])
                    freq = self._mode_freq.get(s["mode"])
                    s["label"] = f"SC {sc}, mode {s['mode']}" + (f" ({freq:.4g} Hz)" if freq else "")
                elif kind == "frequency":
                    s["label"] = f"SC {sc}, f = {s['x']:.5g} Hz"
                elif kind == "time":
                    s["label"] = f"SC {sc}, t = {s['x']:.5g}"
                else:
                    s["label"] = f"SC {sc}"
        steps.sort(key=lambda s: (s["subcase"], s["family"], s["x"], s["domain"]))
        self.steps = steps
        self.domains = {s["domain"]: s["label"] for s in steps}

    def step_axis(self, step=0):
        if not self.steps:
            return ""
        return {"frequency": "Frequency [Hz]", "time": "Time", "mode": "Mode"}.get(self.steps[step]["kind"], "Step")

    # ---------------------------------------------------------------------------------------------- quantities
    def _quantity(self, name, location, group, item, **definition):
        while name in self.result_meta:                         # e.g. real and complex table of the same result
            name += " *"
        self._quantities[name] = definition
        self._register(name, location, group, item)

    def _register_results(self):
        nodal = {k: i for k, i in self._tables.items() if i["kind"] == "nodal"}
        first = ("DISPLACEMENT", "EIGENVECTOR", "VELOCITY", "ACCELERATION")
        order = [k for b in first for k, i in nodal.items() if i["base"] == b]

        def add_nodal(key):
            info = self._tables[key]
            label = self._nodal_label(info["base"])
            comps = info["comps"]
            domains = list(info["blocks"])
            samples = [domains[0], domains[len(domains) // 2], domains[-1]]
            for group, mag_name in ((("X", "Y", "Z"), "magnitude"), (("RX", "RY", "RZ"), "rot. magnitude")):
                present = [c for c in group if c in comps]
                cols = [comps.index(c) for c in present]
                if not present:
                    continue
                if mag_name.startswith("rot") and not any(np.any(np.nan_to_num(self._nodal_array(key, d)[:, cols]))
                                                          for d in samples):
                    continue                                    # rotations of a solid model are zero -> not shown
                self._quantity(f"{label} {mag_name}", "point", label, mag_name, source="nodal", table=key,
                               cols=cols, op="magnitude")
                for c, col in zip(present, cols):
                    self._quantity(f"{label} {c}", "point", label, c, source="nodal", table=key, cols=[col],
                                   op="component")

        for key in order:
            add_nodal(key)
        for base in ("STRESS", "STRAIN"):
            for cplx in (False, True):
                if not any(i["kind"] == "element" and i["base"] == base and i["complex"] == cplx
                           for i in self._tables.values()):
                    continue
                label = base.capitalize()
                for where, location, suffix in (("cell", "cell", "(element)"), ("node", "point", "(nodal avg.)")):
                    group = f"{label} {suffix}"
                    common = dict(source="element", base=base, complex=cplx, where=where)
                    if base == "STRESS" and not cplx:
                        self._quantity(f"{label} von Mises {suffix}", location, group, "von Mises", op="von_mises",
                                       **common)
                        self._quantity(f"{label} max principal {suffix}", location, group, "max principal",
                                       op="max_principal", **common)
                        self._quantity(f"{label} min principal {suffix}", location, group, "min principal",
                                       op="min_principal", **common)
                    for k, c in enumerate(("XX", "YY", "ZZ", "XY", "YZ", "ZX")):
                        self._quantity(f"{label} {c} {suffix}", location, group, c, op="component", col=k, **common)
        for key in nodal:
            if key not in order:
                add_nodal(key)

    def is_complex(self, name):
        q = self._quantities.get(name)
        if q is None:
            return False
        return self._tables[q["table"]]["complex"] if q["source"] == "nodal" else q["complex"]

    # ---------------------------------------------------------------------------------------------- reading
    def _cache_get(self, key):
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key, value):
        size = lambda v: sum(a.nbytes for a in v.values()) if isinstance(v, dict) else getattr(v, "nbytes", 0)
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > 1 and sum(size(v) for v in self._cache.values()) > self._CACHE_BYTES:
                self._cache.popitem(last=False)
        return value

    def _read_rows(self, key, rows):
        """Rows of a table: (start, stop) or array of row numbers."""
        with HDF5_LOCK:
            tbl = self._file().get_node(self._tables[key]["path"])
            if isinstance(rows, tuple):
                return tbl.read(*rows)
            return tbl.read_coordinates(np.asarray(rows, dtype=np.int64))

    def _nodal_array(self, key, domain, cache=True):
        """Nodal table of one domain -> array (n_nodes, n_components) in node order (complex for *_CPLX).
        cache=False -> the array is not stored in the cache (sweeps over all domains, e.g. random response)."""
        cached = self._cache_get(("nodal", key, domain))
        if cached is not None:
            return cached
        info = self._tables[key]
        block = info["blocks"].get(domain)
        if block is None:
            return None
        rows = self._read_rows(key, block)
        idx = self.node_index(rows["ID"])
        ok = idx >= 0
        if info["complex"]:
            values = np.column_stack([rows[c + "R"] + 1j * rows[c + "I"] for c in info["comps"]])
        else:
            values = np.column_stack([rows[c] for c in info["comps"]]).astype(float)
        array = np.full((len(self.node_ids), len(info["comps"])), np.nan, dtype=values.dtype)
        array[idx[ok]] = values[ok]
        return self._cache_put(("nodal", key, domain), array) if cache else array

    def _element_values(self, rows, cplx):
        """Element table rows -> (GRID ids (n, positions), values (n, positions, 6))."""
        n = len(rows)
        grid_ids = rows["GRID"].reshape(n, -1)
        if cplx:
            values = np.stack([(rows[c + "R"] + 1j * rows[c + "I"]).reshape(n, -1) for c in self._SOLID_COMPONENTS],
                              axis=-1)
        else:
            values = np.stack([rows[c].reshape(n, -1) for c in self._SOLID_COMPONENTS], axis=-1).astype(float)
        return grid_ids, values

    @staticmethod
    def _centroid(grid_ids, values):
        """Position 0 = centroid (GRID = 0), next positions = corner nodes."""
        if grid_ids.shape[1] > 1 and np.all(grid_ids[:, 0] == 0):
            return values[:, 0, :], grid_ids[:, 1:], values[:, 1:, :]
        return values.mean(axis=1), grid_ids, values

    def _element_arrays(self, base, cplx, domain, cache=True):
        """Element results of one domain -> {"cell": (n_cells, 6), "node": (n_nodes, 6) nodal averaged}."""
        cache_key = ("element", base, cplx, domain)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        keys = [k for k, i in self._tables.items() if i["kind"] == "element" and i["base"] == base
                and i["complex"] == cplx and domain in i["blocks"]]
        if not keys:
            return None
        n_cells, n_nodes = len(self.element_ids), len(self.node_ids)
        dtype = complex if cplx else float
        cell = np.full((n_cells, 6), np.nan, dtype=dtype)
        total = np.zeros((n_nodes, 6), dtype=dtype)
        count = np.zeros(n_nodes)
        for key in keys:
            rows = self._read_rows(key, self._tables[key]["blocks"][domain])
            centroid, corner_ids, corner_values = self._centroid(*self._element_values(rows, cplx))
            cidx = self.cell_index(rows["EID"])
            cell[cidx[cidx >= 0]] = centroid[cidx >= 0]
            nidx = self.node_index(corner_ids.ravel())
            ok = nidx >= 0
            corner_values = corner_values.reshape(-1, 6)[ok]
            for k in range(6):
                total[:, k] += np.bincount(nidx[ok], weights=corner_values[:, k].real, minlength=n_nodes)
                if cplx:
                    total[:, k] += 1j * np.bincount(nidx[ok], weights=corner_values[:, k].imag, minlength=n_nodes)
            count += np.bincount(nidx[ok], minlength=n_nodes)
        with np.errstate(invalid="ignore", divide="ignore"):
            node = total / count[:, None]
        node[count == 0] = np.nan
        self._fill_midside_nodes(node)
        result = {"cell": cell, "node": node}
        return self._cache_put(cache_key, result) if cache else result

    def _derive(self, s, q, mode):
        """Stress / strain components (n, 6) -> result of the quantity."""
        op = q["op"]
        if op == "von_mises":
            return self.von_mises(s)
        if op == "max_principal":
            return self.principal(s)[:, 2]
        if op == "min_principal":
            return self.principal(s)[:, 0]
        return self.complex_part(s[:, q["col"]], mode)

    def values(self, name, step=0, mode="amplitude"):
        if name not in self._quantities:
            return super().values(name, step, mode)
        q = self._quantities[name]
        n = len(self.node_ids) if self.result_location(name) == "point" else len(self.element_ids)
        if not self.steps:
            return np.full(n, np.nan)
        domain = self.steps[step]["domain"]
        mode = mode if self.is_complex(name) else "real"
        cache_key = ("values", name, domain, mode)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if q["source"] == "nodal":
            array = self._nodal_array(q["table"], domain)
            if array is None:
                result = np.full(n, np.nan)
            elif q["op"] == "magnitude":
                result = self.vector_magnitude(array[:, q["cols"]], mode)
            else:
                result = self.complex_part(array[:, q["cols"][0]], mode)
        else:
            arrays = self._element_arrays(q["base"], q["complex"], domain)
            result = np.full(n, np.nan) if arrays is None else self._derive(arrays[q["where"]], q, mode)
        return self._cache_put(cache_key, np.asarray(result, dtype=float))

    def deformation_vectors(self, step=0):
        if not self.steps:
            return None
        domain = self.steps[step]["domain"]
        for base in ("DISPLACEMENT", "EIGENVECTOR"):
            for key, info in self._tables.items():
                if info["kind"] != "nodal" or info["base"] != base or domain not in info["blocks"]:
                    continue
                if not all(c in info["comps"] for c in ("X", "Y", "Z")):
                    continue
                array = self._nodal_array(key, domain)[:, [info["comps"].index(c) for c in ("X", "Y", "Z")]]
                return np.nan_to_num(array)
        return None

    def get_deformation(self, step=0, phase=0.0):
        array = self.deformation_vectors(step)
        if array is not None and np.iscomplexobj(array):
            array = (array * np.exp(1j * np.radians(phase))).real
        return array

    # ---------------------------------------------------------------------------------------------- history
    def _row_map(self, key, domain):
        """Sorted IDs of the rows of a table block + their order (to find the row of one node / element)."""
        with HDF5_LOCK:
            return self._row_map_locked(key, domain)

    def _row_map_locked(self, key, domain):
        info = self._tables[key]
        if "same_order" not in info:                            # do all blocks list the IDs in the same order?
            blocks = list(info["blocks"].items())
            same = all(isinstance(b, tuple) for _, b in blocks)
            if same and len(blocks) > 1:
                lengths = {b[1] - b[0] for _, b in blocks}
                same = len(lengths) == 1
                if same:
                    field = info["id_field"]
                    first = self._read_rows(key, blocks[0][1])[field]
                    for _, b in (blocks[1], blocks[len(blocks) // 2], blocks[-1]):
                        if not np.array_equal(first, self._read_rows(key, b)[field]):
                            same = False
                            break
            info["same_order"] = same
            info["first_domain"] = blocks[0][0]
        if info["same_order"]:
            domain = info["first_domain"]
        if (key, domain) not in self._row_maps:
            ids = np.asarray(self._read_rows(key, info["blocks"][domain])[info["id_field"]], dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            self._row_maps[(key, domain)] = (ids[order], order)
        return self._row_maps[(key, domain)]

    def _rows_of_id(self, key, domains, id_value):
        """Row of node / element `id_value` in table `key` for every domain (None where missing)."""
        info = self._tables[key]
        coords = []
        for d in domains:
            block = info["blocks"].get(d)
            if block is None:
                coords.append(None)
                continue
            ids, order = self._row_map(key, d)
            pos = np.searchsorted(ids, id_value)
            if pos >= len(ids) or ids[pos] != id_value:
                coords.append(None)
                continue
            row = order[pos]
            coords.append(block[0] + row if isinstance(block, tuple) else int(block[row]))
        valid = [c for c in coords if c is not None]
        if not valid:
            return [None] * len(domains)
        unique = np.unique(valid)
        rows = self._read_rows(key, unique)
        lookup = {int(c): rows[i:i + 1] for i, c in enumerate(unique)}       # 1-row arrays
        return [lookup[int(c)] if c is not None else None for c in coords]

    def _cells_of_node(self, node):
        """Cells which have the node as a corner node."""
        if self._adjacent is None:                              # numpy only (no VTK - also for threads)
            conns, cells, offset = [], [], 0
            for _, block in self._linear_blocks():
                n, k = block.shape
                conns.append(block.ravel())
                cells.append(np.repeat(np.arange(offset, offset + n), k))
                offset += n
            conn, cells = np.concatenate(conns), np.concatenate(cells)
            order = np.argsort(conn, kind="stable")
            self._adjacent = (conn[order], cells[order])
        conn, cells = self._adjacent
        a, b = np.searchsorted(conn, node), np.searchsorted(conn, node, side="right")
        return cells[a:b]

    def _element_history(self, q, index, domains):
        """Stress / strain components (n_domains, 6) of one element (centroid) or one node (averaged)."""
        cplx, base = q["complex"], q["base"]
        dtype = complex if cplx else float
        table_of = {i["element"]: k for k, i in self._tables.items()
                    if i["kind"] == "element" and i["base"] == base and i["complex"] == cplx}

        def element_rows(cell):
            key = table_of.get(self.element_types[cell])
            if key is None:
                return [None] * len(domains)
            return self._rows_of_id(key, domains, int(self.element_ids[cell]))

        if q["where"] == "cell":
            result = np.full((len(domains), 6), np.nan, dtype=dtype)
            for i, row in enumerate(element_rows(index)):
                if row is not None:
                    centroid, _, _ = self._centroid(*self._element_values(row, cplx))
                    result[i] = centroid[0]
            return result

        def node_history(node):
            node_id = int(self.node_ids[node])
            total = np.zeros((len(domains), 6), dtype=dtype)
            count = np.zeros(len(domains))
            for cell in self._cells_of_node(node):
                for i, row in enumerate(element_rows(cell)):
                    if row is None:
                        continue
                    grid_ids, values = self._element_values(row, cplx)
                    pos = np.flatnonzero(grid_ids[0] == node_id)
                    if pos.size:
                        total[i] += values[0, pos[0]]
                        count[i] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = total / count[:, None]
            avg[count == 0] = np.nan
            return avg

        if len(self._cells_of_node(index)):
            return node_history(index)
        for vtk_type, conn in self._cell_blocks:                 # midside node -> average of the edge nodes
            for mid, a, b in self._QUADRATIC_CELLS.get(vtk_type, (None, None, []))[2]:
                rows = np.flatnonzero(conn[:, mid] == index)
                if rows.size:
                    return 0.5 * (node_history(conn[rows[0], a]) + node_history(conn[rows[0], b]))
        return np.full((len(domains), 6), np.nan, dtype=dtype)

    def series_steps(self, step=0):
        """Indices of the steps of the same course as `step` (time steps / frequencies / modes of one subcase)."""
        if not self.steps:
            return []
        series = self.steps[step]["series"]
        return [i for i, s in enumerate(self.steps) if s["series"] == series]

    def nodal_history(self, key, index, domains):
        """Components (n_domains, n_components) of nodal table `key` in node `index` (complex for *_CPLX)."""
        info = self._tables[key]
        dtype = complex if info["complex"] else float
        data = np.full((len(domains), len(info["comps"])), np.nan, dtype=dtype)
        for i, row in enumerate(self._rows_of_id(key, domains, int(self.node_ids[index]))):
            if row is not None:
                if info["complex"]:
                    data[i] = [row[c + "R"][0] + 1j * row[c + "I"][0] for c in info["comps"]]
                else:
                    data[i] = [row[c][0] for c in info["comps"]]
        return data

    def history(self, name, index, step=0, mode="amplitude"):
        if name not in self._quantities or not self.steps:
            return None
        selected = self.series_steps(step)
        if len(selected) < 2:
            return None
        q = self._quantities[name]
        mode = mode if self.is_complex(name) else "real"
        domains = [self.steps[i]["domain"] for i in selected]
        x = np.array([self.steps[i]["x"] for i in selected])
        if q["source"] == "nodal":
            data = self.nodal_history(q["table"], index, domains)
            if q["op"] == "magnitude":
                y = self.vector_magnitude(data[:, q["cols"]], mode)
            else:
                y = self.complex_part(data[:, q["cols"][0]], mode)
        else:
            y = self._derive(self._element_history(q, index, domains), q, mode)
        return np.array(selected), x, np.asarray(y, dtype=float)


if __name__ == "__main__":
    # Run this file directly -> opens the 3D viewer of the H5 file below (path is relative to this file,
    # so it does not depend on the working directory).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "visualization" package
    from visualization.viewer_window import open_viewer

    # the file is loaded in the background (progress in the status bar), other files: File > Open / drag & drop
    open_viewer(Path(__file__).resolve().parent / "box2_weight1g_z_ch.h5")
