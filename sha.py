from afx import *


class SimpleShader(tl.DataOriented):
    @ti.func
    def fallback(self, r):
        t = 0.5 * r.dir.y + 0.5
        blue = V(0.5, 0.7, 1.0)
        white = V(1.0, 1.0, 1.0)
        ret = (1 - t) * white + t * blue
        ret *= 0.1
        if r.dir.y > 0.9:
            ret = 5
        return ret

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
        r.org += r.dir * h.depth

        rspec = tl.reflect(r.dir, h.nrm)
        rdiff = tangent(h.nrm) @ spherical(ti.random(), ti.random())

        kd = 1
        r.dir = tl.normalize((1 - kd) * rspec + kd * rdiff)
        r.org += h.nrm * EPS * 2
        return r
