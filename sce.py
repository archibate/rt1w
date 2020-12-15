from afx import *
from hit import *
from tre import *


class SphereScene(tl.DataOriented):
    def __init__(self):
        self.objs = Sphere.field(32)

        @ti.materialize_callback
        @ti.kernel
        def init_objs():
            for i in self.objs:
                self.objs[i] = Sphere(tl.randNDRange(V3(-1), V3(1)), 0.2)

        self.tree = Octree()

    def build_tree(self):
        self.tree.build(self)

    @ti.func
    def intersect(self, r):
        return self.tree.intersect(self, r)

    @ti.func
    def aintersect(self, r):
        ret = Hit.empty()
        for i in range(self.objs.shape[0]):
            h = self.objs[i].intersect(r)
            ret = ret.union(h)
        return ret

class MeshScene(tl.DataOriented):
    def __init__(self, verts):
        self.objs = Triangle.field(len(verts))

        @ti.materialize_callback
        def init_objs():
            self.objs.v0.from_numpy(verts[:, 0])
            self.objs.v1.from_numpy(verts[:, 1])
            self.objs.v2.from_numpy(verts[:, 2])

        self.tree = Octree()

    def build_tree(self):
        self.tree.build(self)

    @ti.func
    def intersect(self, r):
        return self.tree.intersect(self, r)

    @ti.func
    def aintersect(self, r):
        ret = Hit.empty()
        for i in range(self.objs.shape[0]):
            h = self.objs[i].intersect(r)
            ret = ret.union(h)
        return ret


class SimpleShader(tl.DataOriented):
    @ti.func
    def fallback(self, r):
        t = 0.5 * r.dir.y + 0.5
        blue = V(0.0, 0.0, 0.0)#0.5, 0.7, 1.0)
        white = V(1.0, 1.0, 1.0)
        return (1 - t) * white + t * blue

    @ti.func
    def transmit(self, r, h):
        if not h.is_hit():
            r.color *= self.fallback(r)
            r.kill()
        else:
            r = self._transmit(r, h)
        return r

    @ti.func
    def _transmit(self, r, h):
        r.dir = tl.reflect(r.dir, h.nrm)
        r.org += h.nrm * EPS * 2
        return r
