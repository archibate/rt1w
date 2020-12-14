__import__('sys').path.insert(0, '/home/bate/Develop/glsl_taichi')

import numpy as np
import taichi as ti
import taichi_glsl as tl
ts = tl


V = lambda *_: tl.vec(*_).cast(float)
V3 = lambda *_: tl.vec3(*_).cast(float)
V2 = lambda *_: tl.vec2(*_).cast(float)


EPS = 1e-6
INF = 1e6
