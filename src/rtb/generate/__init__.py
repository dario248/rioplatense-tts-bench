"""Stage 1 — generation.

Importing this subpackage must stay cheap and GPU-free: adapters import their heavy
dependencies inside `load()`, never at module level, and the runner resolves adapter classes
with importlib only when it is about to use one (SPEC.md section 3).
"""
