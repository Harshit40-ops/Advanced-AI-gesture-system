"""Concrete system-action implementations (volume, media, mouse, etc.).

Functions here are plain callables with no side effects on import. Each returns
a short human-readable status string so the dispatcher can log / display what
happened. OS-specific code is isolated behind small helpers so non-Windows
support can be added later.
"""
