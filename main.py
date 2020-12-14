from afx import *
from ray import *
from hit import *


class Scene(tl.DataOriented):
    def __init__(self):
        self.objs = Sphere.field(32)

        @ti.materialize_callback
        @ti.kernel
        def init_objs():
            for i in self.objs:
                self.objs[i] = Sphere(tl.randNDRange(V3(-1), V3(1)), 0.1)

    @ti.func
    def intersect(self, r):
        ret = Hit.empty()
        for i in range(self.objs.shape[0]):
            h = self.objs[i].intersect(r)
            ret = ret.union(h)
        return ret


class Engine(tl.DataOriented):
    def __init__(self, scene, res=512, nsamps=4):
        if isinstance(res, int):
            res = res, res
        nrays = res[0] * res[1] * nsamps

        self.res = ti.Vector(res)
        self.nrays = nrays

        self.rays = Ray.field(nrays)
        self.image = ti.Vector.field(3, float, res)
        self.count = ti.field(int, res)

        self.scene = scene

    @ti.kernel
    def load(self):
        for I in ti.grouped(self.image):
            uv = (I + tl.randND(2)) / self.res * 2 - 1
            org = V(0, 0, -3)
            dir = V(uv, 2).normalized()
            color = V(1, 1, 1)

            r = Ray(org, dir, I, color)
            i = tl.vec(1, self.res.x).dot(I)
            self.rays[i] = r

    @ti.func
    def transmit(self, r):
        h = self.scene.intersect(r)
        if not h.is_hit():
            r.color = 0
        else:
            r.color *= h.nrm
        return r

    @ti.kernel
    def step(self):
        for i in self.rays:
            r = self.rays[i]
            r = self.transmit(r)
            self.rays[i] = r

    @ti.kernel
    def back(self):
        for i in self.rays:
            r = self.rays[i]
            self.count[r.coord] += 1
            self.image[r.coord] += r.color

    @ti.kernel
    def normalize_image(self):
        for I in ti.grouped(self.image):
            self.image[I] /= self.count[I]

    def main(self):
        self.load()
        self.step()
        self.back()
        self.normalize_image()
        ti.imshow(self.image)


s = Scene()
e = Engine(s)
e.main()
