from afx import *
from ray import *
from hit import *
from sce import *
import ezprof


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
            org = V(0, 0, 3)
            dir = V(uv, -2).normalized()
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
            if not r.is_dead():
                r = self.transmit(r)
                self.rays[i] = r

    @ti.kernel
    def back(self):
        for i in self.rays:
            r = self.rays[i]
            if r.is_dead():
                self.count[r.coord] += 1
                self.image[r.coord] += r.color

    @ti.kernel
    def normalize_image(self):
        for I in ti.grouped(self.image):
            self.image[I] /= self.count[I]

    def main(self, ntimes=4, nsteps=3):
        with ezprof.scope('build'):
            self.scene.build_tree()
        for t in range(ntimes):
            with ezprof.scope('time'):
                self.load()
                print(f'\rRendering {100 * t / ntimes:4.01f}%...', end='')
                for s in range(nsteps):
                    self.step()
                self.back()
        print('done')
        with ezprof.scope('norm'):
            self.normalize_image()
            img = self.image.to_numpy()
        ezprof.show()
        ti.imshow(img)


from taichi_three import readobj
obj = readobj('/home/bate/Develop/three_taichi/assets/monkey.obj')
verts = obj['vp'][obj['f'][:, :, 0]]

ti.init(ti.cpu, cpu_max_num_threads=1)
s = MeshScene(verts)
t = SimpleShader()
e = Engine(s, t)
e.main()
