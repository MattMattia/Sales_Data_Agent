"""
sandbox.py - Restricted execution environment for LLM-generated chart code.

Defense-in-depth layers:
1. Static AST validation: import whitelist, forbidden builtins, forbidden
   attribute names (dunders, private members, OS/process escape hatches).
2. Restricted `__builtins__` with a whitelist-checking `__import__`, plus a
   headless matplotlib backend ("Agg") so rendering never touches a GUI thread.
3. Process isolation: validated code runs in a short-lived child interpreter
   with an allowlisted environment (no API keys, passwords or tokens), a hard
   timeout, and a throwaway working directory. Infinite loops or crashes never
   reach the Streamlit process or the memory that holds its credentials, and
   files written through relative paths are discarded; only the chart images
   declared through savefig() are copied back to the app's working directory.

The child process is not a container: absolute paths and the network are
still reachable if a library writer slips past layer 1. Run the app itself in
the provided Docker image for internet-facing deployments.
"""

import ast
import builtins
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("sales_data_agent.sandbox")

ALLOWED_MODULES = frozenset({
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "pandas",
    "numpy",
    "math",
    "statistics",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
})

FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "input", "breakpoint", "exit",
    "quit", "help", "memoryview", "dir",
})

# Attribute names that reach the OS, the interpreter, or file/process APIs
# through otherwise-allowed libraries (e.g. `pd.io.common.os.system`).
FORBIDDEN_ATTRIBUTES = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "importlib", "builtins",
    "ctypes", "ctypeslib", "pickle", "marshal", "tempfile", "glob", "pathlib",
    "system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve",
    "execl", "fork", "kill", "remove", "unlink", "rmdir", "rmtree",
    "environ", "getenv", "putenv", "load_library", "LoadLibrary",
    # Library-level file readers/writers that bypass the `open` ban
    # (e.g. `pd.read_csv('.env')` would leak secrets from the working directory).
    "io", "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap",
    "imread", "DataSource",
    # Library-level file writers that always take a destination (charts may
    # only be written through savefig).
    "save", "savez", "savez_compressed", "savetxt", "tofile", "imsave", "dump",
    "to_excel", "to_parquet", "to_pickle", "to_sql", "to_hdf", "to_feather",
    "to_stata", "to_orc", "to_clipboard", "to_gbq",
    # to_xml(stylesheet=...) reads a local file or URL even without a destination.
    "to_xml",
    "backends", "PdfPages", "ExcelWriter", "HDFStore", "lib", "open_memmap",
})

# Attribute prefixes for file/URL readers (pandas `read_*`) and matplotlib
# canvas writers (`print_figure`, `print_png`, ...) that bypass savefig.
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_", "print_")

# The only file a chart may write: a literal, relative image path inside the working directory.
CHART_SAVE_FUNCTIONS = frozenset({"savefig"})
ALLOWED_CHART_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg")

# pandas formatters that return a string when called without a destination but
# write a file when given one: allowed only as direct calls with no path.
PATH_OPTIONAL_WRITERS = frozenset({
    "to_csv", "to_json", "to_html", "to_latex", "to_markdown", "to_string",
})
_DESTINATION_KEYWORDS = frozenset({"path_or_buf", "path_or_buffer", "buf", "path", "fname", "excel_writer"})
_GUARDED_CALLABLES = CHART_SAVE_FUNCTIONS | PATH_OPTIONAL_WRITERS

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "StopIteration",
)

BLOCKED_PREFIX = "Python execution blocked by security policy"

DEFAULT_TIMEOUT_SECONDS = 15.0

# Only these variables reach the child interpreter (allowlist, not denylist).
CHILD_ENV_ALLOWLIST = frozenset({
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR",
    "HOME", "USERPROFILE", "LANG", "LC_ALL",
})
_SECRET_NAME_RE = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)
_SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))

# Executed by the child with `python -I -c`: the code arrives on stdin and the
# child re-applies layer 2 (restricted builtins + Agg backend) before exec().
_CHILD_RUNNER = (
    "import sys\n"
    f"sys.path.insert(0, {_SANDBOX_DIR!r})\n"
    "import sandbox\n"
    "sys.stdout.write(sandbox._execute_in_process(sys.stdin.read()))\n"
)


class SandboxViolation(Exception):
    """Raised when submitted code violates the sandbox policy."""


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        # Guarded references (savefig, path-optional writers) that are the callee
        # of a validated call. ast visits a Call before its `func` child, so the
        # callee is registered here before visit_Name/visit_Attribute see it; any
        # other reference (alias, argument, return value) would bypass the check.
        self._checked_refs: set = set()
        self.chart_targets: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            _check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise SandboxViolation("relative imports are not allowed")
        _check_module(node.module or "")
        for alias in node.names:
            _check_attribute_name(alias.name)
            if alias.name in _GUARDED_CALLABLES and alias.asname not in (None, alias.name):
                raise SandboxViolation(f"renaming '{alias.name}' on import is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_CALLS or node.id.startswith("__"):
            raise SandboxViolation(f"use of '{node.id}' is not allowed")
        self._check_guarded_reference(node, node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        _check_attribute_name(node.attr)
        self._check_guarded_reference(node, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in CHART_SAVE_FUNCTIONS:
            self.chart_targets.append(_check_chart_save_target(node))
            self._checked_refs.add(id(func))
        elif name in PATH_OPTIONAL_WRITERS:
            _check_pathless_writer(node, name)
            self._checked_refs.add(id(func))
        self.generic_visit(node)

    def _check_guarded_reference(self, node: ast.AST, name: str) -> None:
        if name in _GUARDED_CALLABLES and id(node) not in self._checked_refs:
            rule = "with a literal image path" if name in CHART_SAVE_FUNCTIONS else "without a file destination"
            raise SandboxViolation(f"'{name}' may only be called directly {rule}")


def _check_module(module_name: str) -> None:
    if module_name not in ALLOWED_MODULES:
        raise SandboxViolation(f"import of module '{module_name}' is not allowed")


def _check_attribute_name(attr: str) -> None:
    if attr.startswith("_"):
        raise SandboxViolation(f"access to private/dunder attribute '{attr}' is not allowed")
    if attr in FORBIDDEN_ATTRIBUTES or attr in FORBIDDEN_CALLS or attr.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES):
        raise SandboxViolation(f"access to attribute '{attr}' is not allowed")


def _check_chart_save_target(call: ast.Call) -> str:
    """savefig() must receive a literal, relative image path with no parent traversal; returns it normalized."""
    target = call.args[0] if call.args else next(
        (kw.value for kw in call.keywords if kw.arg == "fname"), None,
    )
    if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
        raise SandboxViolation("savefig() requires a literal relative image path")
    path = target.value.replace("\\", "/")
    parts = path.split("/")
    if path.startswith("/") or ":" in path or ".." in parts or not path.lower().endswith(ALLOWED_CHART_EXTENSIONS):
        raise SandboxViolation(f"savefig() target '{target.value}' must be a relative image path inside the working directory")
    return path


def _check_pathless_writer(call: ast.Call, name: str) -> None:
    """to_csv()/to_json()/... may only return a string: no positional or destination argument."""
    has_destination = bool(call.args) or any(
        kw.arg is None or kw.arg in _DESTINATION_KEYWORDS for kw in call.keywords
    )
    if has_destination:
        raise SandboxViolation(f"{name}() may only be used without a file destination (writing files is not allowed)")


def validate_code_ast(code: str) -> List[str]:
    """
    Parses `code` and raises SandboxViolation if it breaks the policy.
    Returns the relative chart paths the code declares through savefig().
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    visitor = _PolicyVisitor()
    visitor.visit(tree)
    return visitor.chart_targets


def _safe_import(name: str, globals_: Optional[Dict[str, Any]] = None, locals_: Any = None,
                 fromlist: Any = (), level: int = 0) -> Any:
    if level != 0 or name not in ALLOWED_MODULES:
        raise ImportError(f"import of module '{name}' is not allowed in the sandbox")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _restricted_builtins() -> Dict[str, Any]:
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    return safe


def _use_headless_backend() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except ImportError:
        pass


def strip_markdown_fences(code: str) -> str:
    clean = code.strip()
    if clean.startswith("```python"):
        clean = clean[len("```python"):]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def build_child_env(source: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """
    Returns the minimal environment for the sandbox child: allowlisted system
    variables only, never anything whose name looks like a credential.
    """
    source = os.environ if source is None else source
    env = {
        name: value for name, value in source.items()
        if name.upper() in CHILD_ENV_ALLOWLIST and not _SECRET_NAME_RE.search(name)
    }
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def execute_sandboxed(code: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Validates and executes chart/analysis code, returning captured stdout or
    a descriptive error string. Never raises for user code problems.

    The child runs inside a throwaway directory: anything it writes there is
    discarded, except the chart images it declared through savefig(), which
    are copied back to the same relative path in the caller's working dir.
    """
    clean_code = strip_markdown_fences(code)
    try:
        chart_targets = validate_code_ast(clean_code)
    except SandboxViolation as violation:
        return f"{BLOCKED_PREFIX}: {violation}"

    with tempfile.TemporaryDirectory(prefix="sandbox-", ignore_cleanup_errors=True) as workdir:
        for target in chart_targets:
            (Path(workdir) / target).parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _CHILD_RUNNER],
                input=clean_code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=build_child_env(),
                cwd=workdir,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Sandbox child timed out after {timeout_seconds:g}s and was killed.")
            return f"{BLOCKED_PREFIX}: execution timed out after {timeout_seconds:g}s"
        except OSError as exc:
            logger.warning(f"Sandbox subprocess unavailable ({exc}); executing in-process with restricted builtins.")
            return _execute_in_process(clean_code)

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip().splitlines()[-1:] or ["no details"]
            logger.warning(f"Sandbox child exited with code {completed.returncode}: {detail[0]}")
            return f"Python execution error: sandbox process exited with code {completed.returncode}: {detail[0]}"
        _copy_declared_charts(Path(workdir), chart_targets)
        return completed.stdout


def _copy_declared_charts(workdir: Path, chart_targets: List[str]) -> None:
    """Copies each declared chart the child produced back into the current working directory."""
    base = Path.cwd().resolve()
    for target in dict.fromkeys(chart_targets):
        source = workdir / target
        if source.is_symlink() or not source.is_file():
            continue
        destination = (base / target).resolve()
        if base not in destination.parents:
            logger.error(f"Security: refusing to copy chart outside the working directory: {target}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _execute_in_process(clean_code: str) -> str:
    """Layer 2: exec() already-validated code with restricted builtins."""
    _use_headless_backend()
    namespace: Dict[str, Any] = {"__name__": "__sandbox__", "__builtins__": _restricted_builtins()}
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(clean_code, "<sandbox>", "exec"), namespace)  # noqa: S102 - validated above
    except Exception as err:
        return f"Python execution error: {type(err).__name__}: {err}"

    output, errors = stdout_buffer.getvalue(), stderr_buffer.getvalue()
    if output:
        return output
    if errors:
        return f"Executed successfully (warnings: {errors})"
    return "Executed successfully without output."
