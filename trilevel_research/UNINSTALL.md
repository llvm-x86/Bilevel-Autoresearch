# Uninstall tri-level extension

Delete `trilevel_research/`, remove the `trilevel` line from `[project.optional-dependencies]` and `trilevel_research*` from `[tool.setuptools.packages.find]` in `pyproject.toml`, and remove the "Optional: Tri-Level Autoresearch" paragraph from the root `README.md`.

**One-liner for maintainers:**

```bash
rm -rf trilevel_research && git checkout pyproject.toml README.md
```

(Assumes no other edits to those files on the branch.)
