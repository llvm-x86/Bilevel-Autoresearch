from __future__ import annotations

import ast
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class ImportValidation:
    success: bool
    missing_modules: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PreImportValidator:
    """AST-based pre-import validator for generated helper code."""

    def __init__(self) -> None:
        self._approved_packages: List[str] | None = None

    @staticmethod
    def _extract_imports(source_code: str) -> Set[str]:
        """Extract top-level module names from source code AST."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return set()
        modules: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    if top:
                        modules.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    top = node.module.split('.')[0]
                    if top:
                        modules.add(top)
                for alias in node.names:
                    if alias.name and (node.module is None or node.level > 0):
                        top = alias.name.split('.')[0]
                        if top:
                            modules.add(top)
        return modules

    def validate_imports(
        self,
        helper_source: str,
        benchmark_source: str,
    ) -> ImportValidation:
        """Validate that all imports from helper and benchmark are available."""
        result = ImportValidation(success=True)
        try:
            helper_imports = self._extract_imports(helper_source)
            benchmark_imports = self._extract_imports(benchmark_source)
        except Exception as e:
            result.success = False
            result.errors.append(f"AST extraction failed: {e}")
            return result

        all_modules = helper_imports | benchmark_imports
        for module in sorted(all_modules):
            try:
                spec = importlib.util.find_spec(module)
            except (ModuleNotFoundError, ValueError):
                spec = None
            if spec is None:
                result.missing_modules.append(module)
        if result.missing_modules:
            result.success = False
        return result

    def _discover_approved_packages(self) -> List[str]:
        """Scan for pre-installed packages and return as approved list."""
        if self._approved_packages is not None:
            return self._approved_packages

        approved: Set[str] = set()

        # Scan requirements.txt
        req_path = REPO_ROOT / "requirements.txt"
        if req_path.exists():
            try:
                for line in req_path.read_text().splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        # Simple parsing: take the first word/name
                        pkg = stripped.split("==")[0].split(">=")[0].split("<")[0].strip()
                        if pkg:
                            approved.add(pkg.lower())
            except Exception:
                logger.warning("Failed to parse requirements.txt", exc_info=True)

        # Scan pyproject.toml
        pyproject_path = REPO_ROOT / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    tomllib = None  # type: ignore
            if tomllib:
                try:
                    data = tomllib.loads(pyproject_path.read_text())
                    # Look under [project.dependencies] or [build-system] etc.
                    for section_path in (("project", "dependencies"), ("build-system", "requires")):
                        section = data
                        for key in section_path:
                            section = section.get(key, {}) if isinstance(section, dict) else {}
                        if isinstance(section, list):
                            for dep in section:
                                if isinstance(dep, str):
                                    pkg = dep.split("==")[0].split(">=")[0].split("<")[0].strip()
                                    if pkg:
                                        approved.add(pkg.lower())
                except Exception:
                    logger.warning("Failed to parse pyproject.toml", exc_info=True)

        # Add currently loaded top-level modules
        for mod_name in sys.modules:
            top = mod_name.split(".")[0].lower()
            if top:
                approved.add(top)

        self._approved_packages = sorted(approved)
        return self._approved_packages


def pre_import_ast_validator(helper_source: str, benchmark_source: str) -> ImportValidation:
    """Convenience wrapper for one-shot validation."""
    validator = PreImportValidator()
    return validator.validate_imports(helper_source, benchmark_source)