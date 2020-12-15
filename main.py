from afx import *
from ray import *
from hit import *
from sce import *
from sha import *
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
            u, v = (I + tl.randND(2)) / self.res * 2 - 1
            org = V(0, -10, 0)
            dir = V(u, 4, v).normalized()
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
            if not r.is_dead():
                r.color = 0
            self.count[r.coord] += 1
            cnt = self.count[r.coord]
            old = self.image[r.coord]
            self.image[r.coord] = (old * (cnt - 1) + r.color) / cnt

    def main(self, ntimes=12, nsteps=8):
        with ezprof.scope('build'):
            self.scene.build_tree()
        gui = ti.GUI('path', self.image.shape)
        while gui.running:
            if gui.get_event(gui.ESCAPE):
                gui.running = False
            with ezprof.scope('time'):
                self.load()
                for s in range(nsteps):
                    self.step()
                self.back()
            gui.set_image(aces_tonemap(self.image.to_numpy()))
            gui.show()
        gui.close()
        ezprof.show()


if __name__ == '__main__':
    from ldr import readobj, objverts, objmtlids

    #ti.init(ti.cpu, cpu_max_num_threads=1)
    ti.init(ti.cuda)

    if 0:
        scene = SphereScene(2)
    else:
        obj = readobj('assets/cornell.obj')
        obj['v'][:, 2] -= 2
        verts = objverts(obj)
        mids = objmtlids(obj)
        scene = MeshScene(verts, mids)

    shader = SimpleShader()
    engine = Engine(scene, shader)
    engine.main()
