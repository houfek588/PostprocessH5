"""
Main window of the Nastran viewers - results (*.h5, visualization/h5_viewer.py) and input files (*.bdf, *.dat,
*.nas, Preprocess/bdf_viewer.py) are opened in the same window:
    * File > Open... (Ctrl+O), Open recent, Reload (F5), Close file; drag & drop of a file into the window
    * files are loaded in a background thread - progress in the status bar, the window does not freeze
    * File > Save camera view: the view is stored for the file and restored when the file is opened again
Settings (recent files, camera views) are in ~/.nastran_viewer.json.

    from visualization.viewer_window import open_viewer
    open_viewer("results.h5")        # or open_viewer() = empty window with the Open button
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # project root -> packages of the project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.qt_base import (ALIGN_CENTER, USER_ROLE, MainWindow, QtCore, QtGui, QtWidgets,  # noqa: E402
                                   Settings, TaskRunner, _class_enum, qt_app, run_app, safe_call)

H5_SUFFIXES = (".h5",)
BDF_SUFFIXES = (".bdf", ".dat", ".nas", ".blk", ".bulk", ".inc")
FILE_FILTER = ("Nastran files (*.h5 *.bdf *.dat *.nas *.blk *.bulk *.inc);;Results (*.h5);;"
               "Input files (*.bdf *.dat *.nas *.blk *.bulk *.inc);;All files (*)")

DRAG_ENTER = _class_enum(QtCore.QEvent, "Type", "DragEnter")
DRAG_MOVE = _class_enum(QtCore.QEvent, "Type", "DragMove")
DROP = _class_enum(QtCore.QEvent, "Type", "Drop")


def file_kind(path):
    """"h5" = Nastran results, "bdf" = Nastran input file (anything else is read as input file)."""
    return "h5" if Path(path).suffix.lower() in H5_SUFFIXES else "bdf"


def load_model(path, progress=None, verbose=False):
    """Reads a Nastran file: *.h5 -> ParseH5File, other -> BdfModel (can run in a background thread)."""
    if file_kind(path) == "h5":
        from file_processing.read_Nastran_files import ParseH5File
        return ParseH5File(path, verbose=verbose, progress=progress)
    from Preprocess.bdf_model import BdfModel
    return BdfModel(path, verbose=verbose, progress=progress)


def viewer_class(kind):
    """Viewer for a file kind ("h5" / "bdf")."""
    if kind == "bdf":
        from Preprocess.bdf_viewer import BdfViewer
        return BdfViewer
    from visualization.h5_viewer import H5ResultViewer
    return H5ResultViewer


class ViewerWindow(MainWindow):
    """Window with a menu, status bar and one viewer (H5ResultViewer / BdfViewer) or the start page."""

    def __init__(self, window_size=(1650, 980), settings=None, verbose=True):
        qt_app()
        super().__init__()
        self.settings = settings or Settings()
        self.verbose = verbose
        self.viewer = None
        self.loading = None                 # file being loaded
        self._busy = {}                     # running background tasks: name -> (fraction, text)
        self._idle_text = ""
        self._reload_state = None
        self._t0 = 0.0
        self.start_list = None
        self.start_status = None
        self.tasks = TaskRunner()
        self.tasks.on_progress = self.show_task_progress
        self.tasks.on_finished = self.task_finished
        self.resize(*window_size)
        self.setWindowTitle("Nastran viewer")
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_status_bar()
        self._show_start_page()

    # ================================================================================================= menu, status
    def _action(self, menu, text, callback, shortcut=None):
        action = menu.addAction(text)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(lambda checked=False: safe_call(callback))
        return action

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")
        self.open_action = self._action(menu, "&Open...", self.open_dialog, "Ctrl+O")
        self.recent_menu = menu.addMenu("Open &recent")
        self.reload_action = self._action(menu, "Re&load", self.reload, "F5")
        self.close_action = self._action(menu, "&Close file", self.close_file)
        menu.addSeparator()
        self.save_view_action = self._action(menu, "&Save camera view", self.save_view)
        self.restore_view_action = self._action(menu, "Restore saved &view", self.restore_view)
        self.forget_view_action = self._action(menu, "Forget saved view", self.forget_view)
        menu.addSeparator()
        self._action(menu, "E&xit", self.close, "Ctrl+Q")
        self._update_recent()
        self._update_menu()

    def _update_menu(self):
        has_viewer = self.viewer is not None
        saved = has_viewer and self.settings.view(self.viewer.h5.file_path) is not None
        self.reload_action.setEnabled(has_viewer and self.loading is None)
        self.close_action.setEnabled(has_viewer)
        self.save_view_action.setEnabled(has_viewer)
        self.restore_view_action.setEnabled(saved)
        self.forget_view_action.setEnabled(saved)

    def _update_recent(self):
        self.recent_menu.clear()
        files = self.settings.recent_files()
        for k, path in enumerate(files):
            p = Path(path)
            action = self.recent_menu.addAction(f"&{k + 1}  {p.name}    ({p.parent})")
            action.triggered.connect(lambda checked=False, f=path: safe_call(self.open_file, f))
        if files:
            self.recent_menu.addSeparator()
            self._action(self.recent_menu, "Clear list", self._clear_recent)
        else:
            self.recent_menu.addAction("(no recent files)").setEnabled(False)
        self._fill_start_list()

    def _clear_recent(self):
        self.settings.clear_recent()
        self._update_recent()

    def _build_status_bar(self):
        bar = self.statusBar()
        self.status_label = QtWidgets.QLabel()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(240)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.progress_bar)

    def status(self, text):
        """Message in the status bar (shown when no background task is running)."""
        self._idle_text = text
        if not self._busy:
            self.status_label.setText(text)

    def show_task_progress(self, name, fraction, text):
        """Progress of a background task (fraction None = unknown)."""
        self._busy[name] = (fraction, text)
        self._show_busy()

    def task_finished(self, name):
        self._busy.pop(name, None)
        self._show_busy()

    def _show_busy(self):
        if not self._busy:
            self.progress_bar.hide()
            self.status_label.setText(self._idle_text)
            return
        fraction, text = list(self._busy.values())[-1]
        if fraction is None:
            self.progress_bar.setRange(0, 0)                    # busy indicator
        else:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(int(1000 * min(max(fraction, 0.0), 1.0)))
        self.progress_bar.show()
        self.status_label.setText(text)

    # ================================================================================================= start page
    def _show_start_page(self, text=""):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addStretch(1)
        title = QtWidgets.QLabel("<h2>Nastran viewer</h2>Results (*.h5) and input files (*.bdf, *.dat, *.nas)"
                                 "<br><br>Open a file by the button, <b>File &gt; Open</b> (Ctrl+O) or "
                                 "<b>drag &amp; drop</b> it into this window.")
        title.setAlignment(ALIGN_CENTER)
        layout.addWidget(title)
        self.start_status = QtWidgets.QLabel(text)
        self.start_status.setAlignment(ALIGN_CENTER)
        layout.addWidget(self.start_status)
        button = QtWidgets.QPushButton("Open...")
        button.setMinimumWidth(200)
        button.clicked.connect(lambda checked=False: safe_call(self.open_dialog))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.start_list = QtWidgets.QListWidget()
        self.start_list.setMaximumWidth(900)
        self.start_list.setMaximumHeight(260)
        self.start_list.setToolTip("Double-click = open")
        self.start_list.itemActivated.connect(lambda item: safe_call(self.open_file, item.data(USER_ROLE)))
        self.start_recent_label = QtWidgets.QLabel("Recent files:")
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        column = QtWidgets.QVBoxLayout()
        column.addWidget(self.start_recent_label)
        column.addWidget(self.start_list)
        row.addLayout(column, 4)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(2)
        self.setCentralWidget(page)
        self._fill_start_list()

    def _fill_start_list(self):
        if self.start_list is None:
            return
        self.start_list.clear()
        files = self.settings.recent_files()
        for path in files:
            p = Path(path)
            item = QtWidgets.QListWidgetItem(f"{p.name}    -    {p.parent}")
            item.setData(USER_ROLE, path)
            item.setToolTip(path)
            self.start_list.addItem(item)
        self.start_list.setVisible(bool(files))
        self.start_recent_label.setVisible(bool(files))

    # ================================================================================================= open files
    def open_dialog(self):
        recent = self.settings.recent_files()
        start = str(Path(recent[0]).parent) if recent else str(ROOT)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Nastran file", start, FILE_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path):
        """Loads a file in the background and shows it in this window. Returns False if the loading was not
        started. The current file is closed first: VTK objects (mesh, surface) are built by the loading thread and
        VTK must not be used by two threads at the same time (newer VTK versions may deadlock)."""
        path = Path(path)
        if self.loading is not None:
            self.status(f"Wait please - {self.loading.name} is being loaded")
            return False
        if not path.is_file():
            QtWidgets.QMessageBox.warning(self, "Open file", f"File not found:\n{path}")
            return False
        kind = file_kind(path)
        cls = viewer_class(kind)                                # imports in the main thread
        self.loading = path
        self._t0 = time.perf_counter()
        if self.viewer is not None:                             # no rendering while the thread uses VTK
            old, self.viewer = self.viewer, None
            old.dispose()
            self._show_start_page()
        self._update_menu()
        self.setWindowTitle(f"Nastran viewer - loading {path.name}")
        if self.start_status is not None:
            self.start_status.setText(f"<b>Loading {path.name} ...</b>  (progress in the status bar)")
        verbose = self.verbose

        def work(task):
            def report(fraction, text):
                task.progress(fraction, f"{path.name}: {text}")

            model = load_model(path, progress=report, verbose=verbose)
            try:
                prepared = cls.prepare(model, progress=report)
            except BaseException:
                safe_call(getattr(model, "close", lambda: None))
                raise
            return model, prepared

        self.show_task_progress("load", 0.0, f"{path.name}: loading")
        self.tasks.start("load", work, done=lambda result: self._on_loaded(path, cls, *result),
                         error=lambda error: self._on_load_error(path, error))
        return True

    def _on_loaded(self, path, cls, model, prepared):
        self.loading = None
        if hasattr(model, "reopen"):
            model.reopen = False                    # owned by the window: closed for good with the viewer
        try:
            viewer = cls(model, window=self, prepared=prepared)          # -> set_viewer
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            safe_call(getattr(model, "close", lambda: None))
            self._on_load_error(path, error)
            return
        if self.verbose:
            if getattr(model, "report", None):
                print(model.report)                             # BDF: model report (also in the tab Report)
            else:
                safe_call(model.print_summary)
        self.settings.add_recent(path)
        self._update_recent()
        state, self._reload_state = self._reload_state, None
        restored = ""
        if state is not None and Settings.key(state["path"]) == Settings.key(path):
            viewer.set_camera_view(state["view"])
            names = [name for name, _ in viewer.results]
            if state["result"] in names:
                viewer.set_result(names.index(state["result"]))
            if state["step"] < len(viewer.steps):
                viewer.set_step(state["step"])
        else:
            view = self.settings.view(path)
            if view is not None:
                viewer.set_camera_view(view)
                restored = ", saved camera view restored"
        self._update_menu()
        self.status(f"{path.name} loaded in {time.perf_counter() - self._t0:.1f} s{restored}")

    def _on_load_error(self, path, error):
        self.loading = None
        self._reload_state = None
        self.task_finished("load")
        self.setWindowTitle("Nastran viewer")
        if self.start_status is not None:
            self.start_status.setText(f"<span style='color:red'>{path.name} could not be loaded</span>")
        self._update_menu()
        self.status(f"Loading of {path.name} failed")
        lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
        text = lines[-1] if len(lines) > 4 else " ".join(lines)        # e.g. HDF5 back trace -> last line
        QtWidgets.QMessageBox.critical(self, "Open file", f"{path.name} could not be loaded:\n\n"
                                                          f"{type(error).__name__}: {text}\n\n(details in the console)")

    def show_model(self, model):
        """Shows an already loaded model (ParseH5File / BdfModel) - without background loading."""
        kind = "bdf" if type(model).__name__ == "BdfModel" else "h5"
        viewer_class(kind)(model, window=self)
        self.settings.add_recent(model.file_path)
        self._update_recent()
        self._update_menu()

    def set_viewer(self, viewer):
        """Shows a viewer in the window (called by the viewer); the previous viewer is closed."""
        old, self.viewer = self.viewer, viewer
        if old is not None and old is not viewer:
            old.dispose()
        self.start_list = self.start_status = None
        self.setCentralWidget(viewer.widget)
        viewer.pl.interactor.installEventFilter(self)          # drag & drop on the 3D view
        self.setWindowTitle(viewer.title())
        self._update_menu()

    def reload(self):
        """Loads the current file again (e.g. after a new Nastran run), keeps the view, result and step."""
        if self.viewer is None:
            return
        v = self.viewer
        self._reload_state = dict(path=str(v.h5.file_path), view=v.camera_view(), result=v.results[v.idx][0],
                                  step=v.step)
        if not self.open_file(v.h5.file_path):
            self._reload_state = None

    def close_file(self):
        if self.viewer is None:
            return
        old, self.viewer = self.viewer, None
        old.dispose()
        self._show_start_page()
        self.setWindowTitle("Nastran viewer")
        self._update_menu()
        self.status(f"{old.h5.file_path.name} closed")

    # ================================================================================================= camera view
    def save_view(self):
        if self.viewer is None:
            return
        self.settings.set_view(self.viewer.h5.file_path, self.viewer.camera_view())
        self._update_menu()
        self.status(f"Camera view saved for {self.viewer.h5.file_path.name} (restored when the file is opened)")

    def restore_view(self):
        if self.viewer is None:
            return
        view = self.settings.view(self.viewer.h5.file_path)
        if view is not None:
            self.viewer.set_camera_view(view)

    def forget_view(self):
        if self.viewer is not None:
            self.settings.remove_view(self.viewer.h5.file_path)
            self._update_menu()

    # ================================================================================================= drag & drop
    @staticmethod
    def dropped_file(event):
        """First local file with a supported suffix in a drag & drop event (or None)."""
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in H5_SUFFIXES + BDF_SUFFIXES:
                    return path
        return None

    def dragEnterEvent(self, event):  # noqa: N802
        if self.dropped_file(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802
        path = self.dropped_file(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        QtCore.QTimer.singleShot(0, lambda: safe_call(self.open_file, path))   # after the drop is finished

    def eventFilter(self, obj, event):  # noqa: N802
        """Drag & drop on the 3D view (pyvistaqt would read the file as a mesh)."""
        kind = event.type()
        if kind in (DRAG_ENTER, DRAG_MOVE):
            self.dragEnterEvent(event)
            return True
        if kind == DROP:
            self.dropEvent(event)
            return True
        return False

    def closeEvent(self, event):  # noqa: N802
        self.tasks.stop(wait=0.5)
        if self.viewer is not None:
            self.viewer.dispose()
        super().closeEvent(event)


def open_viewer(path=None, model=None, settings=None, verbose=True, window_size=(1650, 980)):
    """
    Opens the viewer window and runs the Qt event loop (returns after the window is closed).

    Args:
        path: file to open (*.h5 results / *.bdf, *.dat, *.nas input), loaded in the background; None = start page
        model: already loaded ParseH5File / BdfModel (instead of path)
    """
    qt_app()
    window = ViewerWindow(window_size=window_size, settings=settings, verbose=verbose)
    window.show()
    if model is not None:
        window.show_model(model)
    elif path is not None:
        window.open_file(path)
    run_app()
    return window
