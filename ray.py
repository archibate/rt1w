from afx import *


class Ray(tl.TaichiClass):
    taichi_class = [
            ('org', ti.Vector.field, 3, float),
            ('dir', ti.Vector.field, 3, float),
            ('coord', ti.Vector.field, 2, int),
            ('color', ti.Vector.field, 3, float),
    ]

    @ti.func
    def is_dead(self):
        return all(self.dir == 0)
