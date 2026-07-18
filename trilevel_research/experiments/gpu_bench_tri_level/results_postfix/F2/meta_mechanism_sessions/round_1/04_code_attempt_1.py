from __future__ import annotations

import ast
import importlib
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _check_imports_ast(code_str: str) -> List[str]:
    """Parse code_str AST, resolve top-level import modules, return missing ones."""
    missing: List[str] = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        logger.warning("AST parse error in generated code: %s", e)
        return ["<syntax_error>"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                try:
                    importlib.import_module(top)
                except ImportError:
                    if top not in missing:
                        missing.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                try:
                    importlib.import_module(top)
                except ImportError:
                    if top not in missing:
                        missing.append(top)

    return missing


def _pre_exec_import_check(
    code_str: str,
    enable_check: bool,
    run_single: callable,
    self_obj,
    *args,
    **kwargs,
):
    """Wrapper that injects AST import check before running generated code.

    If enable_check is True and imports are missing, returns a failure dict.
    Otherwise delegates to original run_single.
    """
    if enable_check:
        missing = _check_imports_ast(code_str)
        if missing:
            logger.warning(
                "Skipping execution – missing import modules: %s", missing
            )
            return {
                "success": False,
                "error": "missing_imports",
                "missing_modules": missing,
            }

    return run_single(self_obj, *args, **kwargs)


# ===== Integration patch =====
def patch_gpu_bench_runner(enable_ast_import_check: bool = True):
    """Monkey-patch GpuBenchRunner with import checking.

    Must be called after GpuBenchRunner is imported.
    """
    try:
        from domains.gpu_bench_opt.mechanism_research import GpuBenchRunner
    except ImportError as e:
        logger.error("Cannot patch – GpuBenchRunner not found: %s", e)
        return

    original_run_single = GpuBenchRunner._run_single

    def patched_run_single(self, *args, **kwargs):
        # Assume the first positional argument is the code_str
        code_str = args[0] if args else kwargs.get("code_str", "")
        return _pre_exec_import_check(
            code_str,
            enable_ast_import_check,
            original_run_single,
            self,
            *args,
            **kwargs,
        )

    GpuBenchRunner._run_single = patched_run_single
    # Also store flag on class for introspection
    GpuBenchRunner._ast_import_check_enabled = enable_ast_import_check
    logger.info(
        "GpuBenchRunner patched: AST import check %s",
        "enabled" if enable_ast_import_check else "disabled",
    )