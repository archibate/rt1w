from afx import *


class Hit(tl.TaichiClass):
    taichi_class = [
            ('depth', ti.field, float),
            ('pos', ti.Vector.field, 3, float),
            ('nrm', ti.Vector.field, 3, float),
            ('tex', ti.Vector.field, 2, float),
            ('id', ti.field, int),
    ]

    @classmethod
    def empty(cls):
        return cls(INF, V3(0), V3(0), V2(0), 0)

    @ti.func
    def union(self, other):
        ret = self
        if ret.depth > other.depth:
            ret = other
        return ret

    @ti.func
    def is_hit(self):
        return self.depth < INF


class Sphere(tl.TaichiClass):
    taichi_class = [
            ('pos', ti.Vector.field, 3, float),
            ('radius', ti.field, float),
            ('id', ti.field, int),
    ]

    @ti.func
    def intersect(self, r):
        t = INF
        op = self.pos - r.org
        b = op.dot(r.dir)
        det = b**2 - op.norm_sqr() + self.radius**2
        if det >= 0:
            det = ti.sqrt(det)
            t = b - det
            if t <= EPS:
                t = b + det
                if t <= EPS:
                    t = INF
        pos = r.org + t * r.dir
        nrm = (pos - self.pos).normalized()
        tex = V(0., 0.)
        return Hit(t, pos, nrm, tex, self.id)

    @ti.func
    def to_bound(self):
        bmin = self.pos - self.radius
        bmax = self.pos + self.radius
        return bmin, bmax


class Triangle(tl.TaichiClass):
    taichi_class = [
            ('v0', ti.Vector.field, 3, float),
            ('v1', ti.Vector.field, 3, float),
            ('v2', ti.Vector.field, 3, float),
            ('id', ti.field, int),
    ]

    @ti.func
    def intersect(self, r):
        ro, rd = r.org, r.dir
        v0, v1, v2 = self.v0, self.v1, self.v2

        e1 = v1 - v0
        e2 = v2 - v0
        p = rd.cross(e2)
        det = e1.dot(p)
        s = ro - v0

        t, u, v = INF, 0.0, 0.0
        ipos, inrm, itex = V(0.0, 0.0, 0.0), V(0.0, 0.0, 0.0), V(0.0, 0.0)

        if det < 0:
            s = -s
            det = -det

        if det >= EPS:
            u = s.dot(p)
            if 0 <= u <= det:
                q = s.cross(e1)
                v = rd.dot(q)
                if v >= 0 and u + v <= det:
                    t = e2.dot(q)
                    det = 1 / det
                    t *= det
                    u *= det
                    v *= det
                    inrm = e2.cross(e1).normalized()
                    ipos = ro + t * rd
                    itex = V(u, v)

        return Hit(t, ipos, inrm, itex, self.id)

    @ti.func
    def to_bound(self):
        bmin = min(self.v0, self.v1, self.v2)
        bmax = max(self.v0, self.v1, self.v2)
        return bmin, bmax


@ti.func
def aabb_hit(bmin, bmax, r):
    near = -INF
    far = INF
    hit = 1

    for i in ti.static(range(3)):
        if abs(r.dir[i]) < EPS:
            if r.org[i] < bmin[i] or r.org[i] > bmax[i]:
                hit = 0
        else:
            i1 = (bmin[i] - r.org[i]) / r.dir[i]
            i2 = (bmax[i] - r.org[i]) / r.dir[i]

            far = min(far, max(i1, i2))
            near = max(near, min(i1, i2))

    if near > far:
        hit = 0

    return hit
