from afx import *
from ray import *
from hit import *
from sce import *


class Engine(tl.DataOriented):
    def __init__(self, scene, shader, res=512, nsamps=4):
        if isinstance(res, int):
            res = res, res
        nrays = res[0] * res[1] * nsamps

        self.res = ti.Vector(res)
        self.nrays = nrays

        self.rays = Ray.field(nrays)
        self.image = ti.Vector.field(3, float, res)
        self.count = ti.field(int, res)

        self.scene = scene
        self.shader = shader

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
        r = self.shader.transmit(r, h)
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
        self.step()
        self.back()
        self.normalize_image()
        ti.imshow(self.image)


s = SphereScene()
t = SimpleShader()
e = Engine(s, t)
e.main()
