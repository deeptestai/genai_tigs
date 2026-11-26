"""Network definitions used by the generators.

Each class keeps the attribute names of the original ``legacy/`` code,
so the published ``.pth`` checkpoints load unchanged. Only inference is
needed here, so training-only helpers (``forward``/``reparameterize``) are omitted.
"""
