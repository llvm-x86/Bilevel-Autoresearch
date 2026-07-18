from dataclasses import dataclass
from typing import Any, Optional
import importlib
import importlib.util
import sys
import traceback

logger = logging.getLogger(__name__)

@dataclass
class ImportResult:
    """Result of a safe import attempt."""
    success: bool
    module: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    traceback_str: Optional[str] = None


class SafeModuleImporter:
    """Wraps importlib in try-except and returns structured ImportResult.
    
    Replaces bare 'importlib.import_module(...)' calls with typed results
    that can be logged and used for meta-learner feedback.
    """
    
    @staticmethod
    def import_module(module_name: str) -> ImportResult:
        """Safely import a module by name.
        
        Args:
            module_name: Fully qualified module name (e.g., 'domains.gpu_bench.runner')
            
        Returns:
            ImportResult with success/failure and error details
        """
        try:
            module = importlib.import_module(module_name)
            return ImportResult(success=True, module=module)
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                f"Safe import failed for module '{module_name}': {e}",
                exc_info=True
            )
            return ImportResult(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                traceback_str=tb
            )
    
    @staticmethod
    def import_from_file(module_name: str, file_path: str) -> ImportResult:
        """Safely import a module from a file path.
        
        Args:
            module_name: Name to give the module (e.g., 'runner_patch')
            file_path: Absolute path to the .py file
            
        Returns:
            ImportResult with success/failure and error details
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return ImportResult(
                    success=False,
                    error_message=f"Could not create spec for {module_name} at {file_path}",
                    error_type="SpecCreationError"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                return ImportResult(success=True, module=module)
            except Exception:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                f"Safe import failed for file '{file_path}' as '{module_name}': {e}",
                exc_info=True
            )
            return ImportResult(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                traceback_str=tb
            )