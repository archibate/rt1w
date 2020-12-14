import sys
sys.path.insert(0, '/home/bate/Develop/glsl_taichi')
import taichi as ti
import taichi_glsl as tl


V = lambda *_: tl.vec(*_).cast(float)
V3 = lambda *_: tl.vec3(*_).cast(float)
V2 = lambda *_: tl.vec2(*_).cast(float)


class Ray(tl.TaichiClass):
    @property
    def org(self):
        return self.entries[0]

    @property
    def dir(self):
        return self.entries[1]

    @property
    def coord(self):
        return self.entries[2]

    @property
    def color(self):
        return self.entries[3]

    @classmethod
    def _field(cls, *_, **__):
        org = ti.Vector.field(3, float, *_, **__)
        dir = ti.Vector.field(3, float, *_, **__)
        coord = ti.Vector.field(2, int, *_, **__)
        color = ti.Vector.field(3, float, *_, **__)
        return org, dir, coord, color

    @ti.func
    def is_dead(self):
        return all(self.dir == 0)


class Engine(tl.DataOriented):
    def __init__(self, res=512, nsamps=4):
        if isinstance(res, int):
            res = res, res
        nrays = res[0] * res[1] * nsamps

        self.res = ti.Vector(res)
        self.nrays = nrays

        self.rays = Ray.field(nrays)
        self.image = ti.Vector.field(3, float, res)
        self.count = ti.field(int, res)

    @ti.kernel
    def load(self):
        for I in ti.grouped(self.image):
            uv = I / self.res
            org = V(0, 0, -2)
            dir = V(uv, 1).normalized()
            color = V(1, 1, 1)

            r = Ray(org, dir, I, color)
            i = tl.vec(1, self.res.x).dot(I)
            self.rays[i] = r

    @ti.func
    def transmit(self, r):
        r.color *= r.dir
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


e = Engine()
e.main()
