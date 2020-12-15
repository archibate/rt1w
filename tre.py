from afx import *
from ray import *
from hit import *


class Array(tl.DataOriented):
    is_taichi_class = True

    @classmethod
    def field(cls, *args, shape):
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape) + 8
        return cls(*args, shape)

    class Proxy(tl.DataOriented):
        is_taichi_class = True

        def __init__(self, parent, I):
            self.parent = parent
            self.I = I

        @ti.func
        def subscript(self, I):
            self.parent[self.I, I]

        def assign(self, rhs):
            assert rhs is None

    @ti.func
    def subscript(self, I):
        return self.Proxy(self, I)



class Octree(tl.DataOriented):
    def __init__(self, N=1024, NE=32):
        self.center = ti.Vector.field(3, float)
        self.size = ti.field(float)
        self.elems = ti.field(int)
        self.children = ti.field(int)
        self.parent = ti.field(int)
        self.count = ti.field(int, ())

        ti.root.dense(ti.i, N).place(self.center, self.size)
        ti.root.dense(ti.i, N).dynamic(ti.j, NE).place(self.elems)
        ti.root.dense(ti.i, N).dense(ti.j, 8).place(self.children)
        ti.root.dense(ti.i, N).place(self.parent)

    @ti.func
    def to_bound(self, i):
        bmin = self.center[i] - self.radius[i]
        bmax = self.center[i] + self.radius[i]
        return bmin, bmax

    @ti.func
    def hit_bound(self, i, r):
        return Sphere(self.center[i], self.size[i]).intersect(r).is_hit()

    @ti.func
    def alloc(self, center, size, parent):
        i = ti.atomic_add(self.count[None], 1)
        self.center[i] = center
        self.size[i] = size
        self.parent[i] = parent
        ti.deactivate(self.elems.parent(), [i])
        for j in range(8):
            self.children[i, j] = 0
        return i

    @ti.func
    def child(self, cur, pos):
        org = self.center[cur]
        x, y, z = 1 if pos >= org else 0
        return z * 4 + y * 2 + x

    @ti.func
    def insert(self, bmin, bmax, id):
        cur = 0
        while True:
            lo = self.child(cur, bmin)
            hi = self.child(cur, bmax)
            if lo != hi:
                break
            if self.children[cur, lo] == 0:
                half = self.size[cur] / 2
                coor = V(lo % 2, lo // 2 % 2, lo // 4)
                pos = self.center[cur] + (half if coor != 0 else -half)
                self.children[cur, lo] = self.alloc(pos, half, cur)
            cur = self.children[cur, lo]
        print('dept', cur)
        ti.append(self.elems.parent(), [cur], id)
        return cur

    @ti.kernel
    def build(self, scene: ti.template()):
        self.count[None] = 0
        self.alloc(V3(0), 2, 0)
        for _ in range(1):
            for i in range(scene.objs.shape[0]):
                obj = scene.objs[i]
                bmin, bmax = obj.to_bound()
                self.insert(bmin, bmax, i)

    @ti.func
    def walk(self, r):
        cur = 0
        for e in range(ti.length(self.elems.parent(), [cur])):
            i = self.elems[cur, e]
            yield i

    @ti.func
    def intersect(self, scene: ti.template(), r):
        ret = Hit.empty()
        for i in ti.static(self.walk(r)):
            h = scene.objs[i].intersect(r)
            ret = ret.union(h)
        return ret
