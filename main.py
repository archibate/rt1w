from afx import *
from ray import *
from hit import *
from sce import *
from sha import *
import ezprof


class Engine(tl.DataOriented):
    def __init__(self, scene, shader, res=512, nsamps=1):
        if isinstance(res, int):
            res = res, res
        self.nsamps = nsamps
        self.res = ti.Vector(res)
        nrays = res[0] * res[1] * nsamps
        self.nrays = nrays

        self.rays = Ray.field(nrays)
        self.image = ti.Vector.field(3, ti.f64, res)
        self.count = ti.field(int, res)

        self.scene = scene
        self.shader = shader

    @ti.kernel
    def load(self):
        for I in ti.grouped(self.image):
            org = V(0, -10, 0)
            i = tl.vec(1, self.res.x).dot(I) * self.nsamps

            for j in range(self.nsamps):
                u, v = (I + tl.randND(2)) / self.res * 2 - 1
                dir = V(u, 4, v).normalized()
                r = Ray(org, dir, I, V3(1))
                self.rays[i + j] = r

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
        for I in ti.grouped(self.image):
            i = tl.vec(1, self.res.x).dot(I) * self.nsamps
            color = V3(0)
            for j in range(self.nsamps):
                r = self.rays[i + j]
                if r.is_dead():
                    color += r.color
            color /= self.nsamps
            self.count[I] += 1
            cnt = self.count[I]
            old = self.image[I]
            self.image[I] = (old * (cnt - 1) + color) / cnt

    def main(self, nsteps=8):
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
