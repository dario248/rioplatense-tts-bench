"""Herramientas que NO son el benchmark: viven afuera de `src/rtb` a proposito.

`tests/test_core_isolation.py` prohibe numpy, soundfile y cualquier libreria pesada adentro del
core. El renderer de video necesita las tres, asi que no puede ser un modulo de `rtb`.
"""
