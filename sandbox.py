"""
sandbox.py - Restricted execution environment for LLM-generated chart code.

Defense-in-depth layers applied before and during `exec()`:
1. Static AST validation: import whitelist, forbidden builtins, forbidden
   attribute names (dunders, private members, OS/process escape hatches).
2. Restricted `__builtins__` with a whitelist-checking `__import__`.
3. Headless matplotlib backend ("Agg") so rendering never touches a GUI thread.

IMPORTANT: an in-process Python sandbox reduces risk but is not a hard
security boundary. For untrusted, internet-facing deployments, run the chart
generator inside an isolated container/process with no secrets in its env.
"""

import ast
import builtins
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

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
})

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "StopIteration",
)

BLOCKED_PREFIX = "Python execution blocked by security policy"


class SandboxViolation(Exception):
    """Raised when submitted code violates the sandbox policy."""


class _PolicyVisitor(ast.NodeVisitor):
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
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_CALLS or node.id.startswith("__"):
            raise SandboxViolation(f"use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        _check_attribute_name(node.attr)
        self.generic_visit(node)


def _check_module(module_name: str) -> None:
    if module_name not in ALLOWED_MODULES:
        raise SandboxViolation(f"import of module '{module_name}' is not allowed")


def _check_attribute_name(attr: str) -> None:
    if attr.startswith("_"):
        raise SandboxViolation(f"access to private/dunder attribute '{attr}' is not allowed")
    if attr in FORBIDDEN_ATTRIBUTES or attr in FORBIDDEN_CALLS:
        raise SandboxViolation(f"access to attribute '{attr}' is not allowed")


def validate_code_ast(code: str) -> None:
    """Parses `code` and raises SandboxViolation if it breaks the policy."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    _PolicyVisitor().visit(tree)


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


def execute_sandboxed(code: str) -> str:
    """
    Validates and executes chart/analysis code, returning captured stdout or
    a descriptive error string. Never raises for user code problems.
    """
    clean_code = strip_markdown_fences(code)
    try:
        validate_code_ast(clean_code)
    except SandboxViolation as violation:
        return f"{BLOCKED_PREFIX}: {violation}"

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
