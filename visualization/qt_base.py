"""
Common Qt parts of the viewers: selection of the Qt binding, QApplication, background tasks, user settings.

Qt for Python is needed (once):   pip install pyvistaqt PySide6        (PyQt5 / PyQt6 work as well)
If more Qt bindings are installed, PySide6 is used (can be changed by the environment variable QT_API).
"""
import importlib.util
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

# PySide6 is preferred (tested; PyQt aborts the whole application on a Python error in a Qt callback)
if not os.environ.get("QT_API") and importlib.util.find_spec("PySide6") is not None:
    os.environ["QT_API"] = "pyside6"

try:
    from pyvistaqt import MainWindow, QtInteractor  # noqa: F401  (re-exported for the viewers)
    from qtpy import QtCore, QtGui, QtWidgets
except Exception as error:  # ImportError or "no Qt bindings found"
    raise ImportError("The viewer needs Qt for Python, install it by:  pip install pyvistaqt PySide6") from error


def _qt_enum(scope, name):
    """Qt enum value working with PySide6 / PyQt5 / PyQt6 (unscoped or scoped access)."""
    value = getattr(QtCore.Qt, name, None)
    return value if value is not None else getattr(getattr(QtCore.Qt, scope), name)


def _class_enum(cls, scope, name):
    """Enum of a Qt class (e.g. QtCore.QEvent.Type.Drop) for PySide6 / PyQt5 / PyQt6."""
    value = getattr(cls, name, None)
    return value if value is not None else getattr(getattr(cls, scope), name)


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


def run_app():
    """Runs the Qt event loop (returns after the last window is closed)."""
    app = qt_app()
    run = getattr(app, "exec", None) or getattr(app, "exec_")
    return run()


USER_ROLE = _qt_enum("ItemDataRole", "UserRole")
HORIZONTAL = _qt_enum("Orientation", "Horizontal")
VERTICAL = _qt_enum("Orientation", "Vertical")
ITEM_ENABLED = _qt_enum("ItemFlag", "ItemIsEnabled")
ITEM_SELECTABLE = _qt_enum("ItemFlag", "ItemIsSelectable")
TEXT_SELECTABLE = _qt_enum("TextInteractionFlag", "TextSelectableByMouse")
ALIGN_CENTER = _qt_enum("AlignmentFlag", "AlignCenter")
ALIGN_RIGHT = _qt_enum("AlignmentFlag", "AlignRight") | _qt_enum("AlignmentFlag", "AlignVCenter")


def safe_call(function, *args):
    """Calls a function from a Qt callback; a Python error is printed (PyQt would abort the application)."""
    try:
        return function(*args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None


# =====================================================================================================================
#   background tasks
# =====================================================================================================================
class Cancelled(Exception):
    """Raised inside a background task after it was cancelled (Task.check / Task.progress)."""


class Task:
    """Handle given to a function running in a background thread."""

    def __init__(self, runner, name, token):
        self._runner = runner
        self.name = name
        self._token = token
        self._cancel = threading.Event()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check(self):
        """Raises Cancelled if the task was cancelled (call it regularly in long loops)."""
        if self._cancel.is_set():
            raise Cancelled()

    def progress(self, fraction=None, text=""):
        """Reports progress (fraction 0..1, None = unknown) shown in the status bar; also checks cancellation."""
        self.check()
        self._runner._queue.put(("progress", self.name, self._token, (fraction, text)))


class TaskRunner:
    """
    Runs functions in background threads so that the window does not freeze. Results, progress and errors are
    delivered to callbacks in the Qt main thread (the queue is polled by a QTimer).

    A new task with the same name supersedes (cancels) the running one - its results are ignored.
    """

    def __init__(self, interval_ms=100):
        self._queue = queue.Queue()
        self._tasks = {}                    # name -> {"task", "thread", "done", "error"}
        self._timer = QtCore.QTimer()
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self.on_progress = None             # optional listener (name, fraction, text) - status bar
        self.on_finished = None             # optional listener (name) - task finished / cancelled

    def start(self, name, function, done=None, error=None):
        """
        Runs function(task) in a background thread.

        Args:
            name (str): task name (a running task with the same name is cancelled)
            function: function(task: Task) -> result; it must not touch Qt widgets
            done: callback(result) in the main thread
            error: callback(exception) in the main thread (the traceback is printed to the console)
        """
        self.cancel(name)
        token = object()
        task = Task(self, name, token)

        def run():
            try:
                result = function(task)
            except Cancelled:
                self._queue.put(("cancelled", name, token, None))
            except BaseException as exc:  # noqa: BLE001
                self._queue.put(("error", name, token, (exc, traceback.format_exc())))
            else:
                self._queue.put(("done", name, token, result))

        thread = threading.Thread(target=run, name=f"viewer-{name}", daemon=True)
        self._tasks[name] = dict(task=task, token=token, thread=thread, done=done, error=error)
        thread.start()
        if not self._timer.isActive():
            self._timer.start()
        return task

    def running(self, name=None):
        return bool(self._tasks) if name is None else name in self._tasks

    def cancel(self, name, wait=0.0):
        """Cancels a task (it stops at its next check / progress call); wait > 0 = wait for the thread [s]."""
        entry = self._tasks.pop(name, None)
        if entry is None:
            return
        entry["task"]._cancel.set()
        if wait > 0:
            entry["thread"].join(wait)
        if self.on_finished is not None:
            safe_call(self.on_finished, name)

    def stop(self, wait=0.0):
        """Cancels all tasks and stops polling."""
        for name in list(self._tasks):
            self.cancel(name, wait)
        self._timer.stop()

    def _poll(self):
        last_progress = {}
        while True:
            try:
                kind, name, token, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            entry = self._tasks.get(name)
            if entry is None or entry["token"] is not token:
                continue                                    # superseded / cancelled task
            if kind == "progress":
                last_progress[name] = payload
                continue
            last_progress.pop(name, None)
            del self._tasks[name]
            if self.on_finished is not None:
                safe_call(self.on_finished, name)
            if kind == "done" and entry["done"] is not None:
                safe_call(entry["done"], payload)
            elif kind == "error":
                exc, text = payload
                print(text, file=sys.stderr)
                if entry["error"] is not None:
                    safe_call(entry["error"], exc)
        if self.on_progress is not None:
            for name, (fraction, text) in last_progress.items():
                if name in self._tasks:
                    safe_call(self.on_progress, name, fraction, text)
        if not self._tasks:
            self._timer.stop()


# =====================================================================================================================
#   user settings (recent files, camera views)
# =====================================================================================================================
class Settings:
    """
    Settings of the viewer stored in a JSON file in the home directory (~/.nastran_viewer.json, or the file given
    by the environment variable NASTRAN_VIEWER_SETTINGS): recent files, saved camera views of the files.
    """

    MAX_RECENT = 10

    def __init__(self, path=None):
        path = path or os.environ.get("NASTRAN_VIEWER_SETTINGS") or Path.home() / ".nastran_viewer.json"
        self.path = Path(path)
        self.data = {"recent": [], "views": {}}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except (OSError, ValueError):
            pass

    def save(self):
        try:
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError as error:
            print(f"Settings were not saved ({self.path}): {error}", file=sys.stderr)

    @staticmethod
    def key(path):
        return os.path.normcase(str(Path(path).resolve()))

    def recent_files(self):
        """Recent files which still exist (newest first)."""
        return [p for p in self.data.get("recent", []) if Path(p).is_file()]

    def add_recent(self, path):
        path = str(Path(path).resolve())
        recent = [p for p in self.data.get("recent", []) if self.key(p) != self.key(path)]
        self.data["recent"] = [path] + recent[:self.MAX_RECENT - 1]
        self.save()

    def clear_recent(self):
        self.data["recent"] = []
        self.save()

    def view(self, path):
        """Saved camera view of a file (dict) or None."""
        return self.data.get("views", {}).get(self.key(path))

    def set_view(self, path, view):
        self.data.setdefault("views", {})[self.key(path)] = view
        self.save()

    def remove_view(self, path):
        if self.data.get("views", {}).pop(self.key(path), None) is not None:
            self.save()
