"""v000 -- identity control. Returns the unmodified baseline.

Not an optimization: this is the harness's calibration run. It should score
~1.00x on every shape. Any systematic deviation means the measurement itself is
biased, and every later speedup would inherit that bias.
"""
def build_model(config, bench):
    return bench.BaselineTransformer(config)
