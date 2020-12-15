from afx import *
from hit import *
from tre import *


class SphereScene(tl.DataOriented):
    def __init__(self, nspheres):
        self.objs = Sphere.field(nspheres)

        @ti.materialize_callback
        @ti.kernel
        def init_objs():
            for i in self.objs:
                self.objs[i] = Sphere(tl.randNDRange(V3(-1), V3(1)), 0.2, i)
            self.objs[0] = Sphere(V(0, +0.5, 0), 1)
            self.objs[1] = Sphere(V(0, -1e2-0.5, 0), 1e2)

        #self.tree = Octree()

    def build_tree(self):
        pass#self.tree.build(self)

    @ti.func
    def tintersect(self, r):
        return self.tree.intersect(self, r)

    @ti.func
    def intersect(self, r):
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
            self.objs.id.from_numpy(np.arange(len(verts)))

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
