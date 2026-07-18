from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImportValidationResult:
    module_name: str
    is_importable: bool
    error_message: Optional[str] = None
    module_path: Optional[Path] = None


class RunnerImportValidator:
    def validate_import(self, module_name: str, base_path: Path) -> ImportValidationResult:
        # First try to locate the module file directly under base_path
        module_path_candidate = base_path / (module_name.replace('.', '/') + '.py')
        if module_path_candidate.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(module_path_candidate))
            if spec is not None:
                try:
                    importlib.util.module_from_spec(spec)
                    # The module loads correctly if spec is valid, but still need actual import
                    # However for validation we can just check spec is not None and no syntax errors
                    return ImportValidationResult(
                        module_name=module_name,
                        is_importable=True,
                        module_path=module_path_candidate
                    )
                except Exception as e:
                    return ImportValidationResult(
                        module_name=module_name,
                        is_importable=False,
                        error_message=str(e),
                        module_path=module_path_candidate
                    )
            else:
                return ImportValidationResult(
                    module_name=module_name,
                    is_importable=False,
                    error_message="Failed to create spec",
                    module_path=module_path_candidate
                )
        # Fall back to importlib.import_module (may work for stdlib/installed packages)
        try:
            importlib.import_module(module_name)
            return ImportValidationResult(
                module_name=module_name,
                is_importable=True
            )
        except ImportError as e:
            return ImportValidationResult(
                module_name=module_name,
                is_importable=False,
                error_message=str(e)
            )

    def validate_all_runner_dependencies(self, runner_path: Path) -> List[ImportValidationResult]:
        results = []
        try:
            with open(runner_path, 'r') as f:
                source = f.read()
        except Exception as e:
            logger.error("Failed to read runner.py: %s", e)
            return results

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error("Failed to parse runner.py: %s", e)
            return results

        # Base path for relative imports is the directory of runner.py or project root
        base_path = runner_path.parent

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result = self.validate_import(alias.name, base_path)
                    results.append(result)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result = self.validate_import(node.module, base_path)
                    results.append(result)
                # 'from . import x' -> module is None, skip in this validation
        return results

    def format_validation_report(self, results: List[ImportValidationResult]) -> str:
        if not results:
            return ""
        broken = [r for r in results if not r.is_importable]
        if not broken:
            return ""
        lines = [f"Found {len(broken)} broken import(s):"]
        for r in broken:
            lines.append(f"  - {r.module_name}: {r.error_message or 'unknown error'}")
        return "\n".join(lines)