## Implementation Specification

### 1. Mechanism name
`pre_import_ast_validator`

### 2. Implementation strategy
`new_helper_class`

### 3. Target
New class `PreImportValidator` in `GpuBenchMechanismResearcher`

### 4. Step-by-step logic

**Step 1: AST-based import extraction**
- Add method `_extract_imports(source_code: str) -> Set[str]` that:
  - Parses source string into AST via `ast.parse()`
  - Walks all nodes with `ast.walk()`
  - For each `ast.Import` node: collect all `alias.name` (e.g., `os`, `numpy.fft`)
  - For each `ast.ImportFrom` node: if module is not None, collect `module` name; also collect `alias.name` from names for relative imports
  - Return set of module strings (top-level package only, e.g., `numpy` from `numpy.fft.fft2`)

**Step 2: Validation method**
- Add method `validate_imports(helper_source: str, benchmark_source: str) -> ImportValidation`:
  - Return dataclass with fields: `success: bool`, `missing_modules: List[str]`, `errors: List[str]`
  - Extract imports from helper source → set A
  - Extract imports from benchmark source → set B  
  - Union = A ∪ B
  - For each module string in union, try `importlib.util.find_spec(module)`
  - If `find_spec` returns None, add to `missing_modules`
  - If any missing, `success = False`

**Step 3: Integration hook in exploration phase**
- After receiving generated helper code from LLM, BEFORE writing file:
  - Call `validate_imports(helper_code, benchmark_code)`
  - If `success = False`, log warning with missing modules
  - If more than 3 missing modules, discard attempt and request regeneration with constraint: "All imports must be from standard library or these pre-approved packages: {json.dumps(approved_packages)}"

**Step 4: Approved package discovery**
- Add method `_discover_approved_packages()` that scans `requirements.txt`, `pyproject.toml`, and current `sys.modules` keys for pre-installed packages
- Cache approved list as instance attribute

### 5. Integration points

- **When**: Called in `_process_explore_generation()` after LLM returns code but before persisting to file
- **What it replaces**: Currently no pre-validation exists; this inserts between generation and file write
- **Logging**: Log missing module names at WARNING level; log validation success at DEBUG level
- **Error recovery**: If validation fails, the researcher can:
  1. Request regeneration with explicit module constraints
  2. Or skip that sample and continue with next
- **Not thread-safe**: Single-threaded validation is fine for current usage pattern

### 6. Optional schedule patch

```json
{
  "level2_interval": 3,
  "enable_pre_import_validation": true,
  "max_missing_modules": 3
}
```