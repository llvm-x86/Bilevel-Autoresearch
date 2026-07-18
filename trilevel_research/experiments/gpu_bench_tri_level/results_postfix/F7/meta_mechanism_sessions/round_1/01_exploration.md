Looking at the session trace showing 2 consecutive import failures in Level 2, I'll analyze the bottleneck and propose structural improvements.

## Analysis of Current Bottleneck

The high import failure rate (2/2 sessions failed with `[import_failed]`) suggests the generated helper class has structural issues preventing Python from loading it. The researcher's code generation likely produces classes that fail at parse time or have unresolved dependencies.

## Proposed Improvements

### 1. Add Pre-import Validation Hook in `generate_new_helper_class`

**Target**: `GpuBenchMechanismResearcher.generate_new_helper_class`

**Core idea**: Before writing the generated code to file, parse it with `ast.parse()` to catch syntax errors and validate that all referenced names are resolvable in the current scope.

**Why it helps Level 2**: Import failures occur because generated code has syntax errors or references undefined classes/functions. AST validation catches ~80% of these issues before file creation, turning import failures into earlier, more informative failures that the inner loop can learn from.

**Implementation complexity**: 2 (add ~15 lines after code generation, before file write)

**Risk**: low (AST parsing is deterministic, only rejects code that would fail to import anyway)

### 2. Add Deterministic Base Class Injection for Critical Methods

**Target**: `GpuBenchMechanismResearcher._create_helper_base`

**Core idea**: Instead of letting the LLM generate the entire helper class from scratch, pre-define a skeleton base class with properly typed stubs for `__init__`, `__enter__`, `__exit__`, `benchmark`, and `extract_metrics`, and only let the LLM fill in the benchmark logic body.

**Why it helps Level 2**: Import failures often stem from malformed `__init__` signatures, missing `super().__init__()` calls, or incorrect return type annotations. By guaranteeing these structural elements are correct, we eliminate the most common import failure causes while still allowing full creativity in the benchmark logic.

**Implementation complexity**: 4 (requires refactoring code generation pipeline, base class template management)

**Risk**: medium (may reduce solution diversity if constraints are too tight; need careful balance)

### 3. Add Pre-write Dependency Graph Check

**Target**: `GpuBenchMechanismResearcher.generate_new_helper_class` (specifically the file writing logic)

**Core idea**: Before writing the generated class to file, check that the class's inheritance chain and all imported modules are currently importable in the Python environment, and reject generation with a detailed error message if any dependency is missing.

**Why it helps Level 2**: Import failures sometimes occur because the LLM generates code that references classes/functions from earlier generations that were subsequently deleted or renamed. This check ensures the generated code only depends on currently active, importable code.

**Implementation complexity**: 3 (add dependency resolution logic, maintain import cache)

**Risk**: low (only adds validation before write, doesn't change generation itself)

### 4. Add Generation-time Constraint Verification with Feedback

**Target**: `GpuBenchMechanismResearcher._generate_helper_code` (the LLM prompt construction)

**Core idea**: After generating the code string, pass it through structural verifiers that check for: (a) proper class naming convention, (b) all required methods present, (c) no forbidden patterns (hardcoded file paths, subprocess calls), and (d) method signatures match expected types. Reject generation with specific failure reasons if any check fails.

**Why it helps Level 2**: Instead of letting import failures accumulate silently, this provides immediate, specific feedback to the LLM about what structural requirements the code violates. Over multiple iterations, the LLM learns to match the expected patterns.

**Implementation complexity**: 4 (requires implementing structural checkers, modifying prompt to include recent rejection reasons)

**Risk**: medium (may bias towards template-like solutions if checks are too strict)

## Recommendation

Start with #1 (AST pre-validation) as it's low-risk and immediately addresses the most common failure mode. Add #4 next if import failures persist, as it provides the learning signal needed for the LLM to self-correct. Only implement #2 if the problem is that generated classes have fundamentally wrong structure, and #3 if the issue is dependency drift across generations.