from afx import *


class SimpleShader(tl.DataOriented):
    @ti.func
    def fallback(self, r):
        t = 0.5 * r.dir.z + 0.5
        blue = V(0.5, 0.7, 1.0)
        white = V(1.0, 1.0, 1.0)
        ret = (1 - t) * white + t * blue
        ret = 0.1
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
        r.dir = r.dir.normalized()
        if r.dir.dot(h.nrm) > 0:
            h.nrm = -h.nrm
        
        r.org += r.dir * h.depth

        rspec = tl.reflect(-r.dir, h.nrm)
        rdiff = tangent(h.nrm) @ spherical(ti.random(), ti.random())

        if h.id == 1:
            r.color *= V(1, .7, .5)
            r.dir = (rspec + 0.06 * rdiff).normalized()

        elif h.id == 2:          
            #if ti.random() >= 0.0:
            #    lipos = V(0, 0, 2)
            #    r.color *= (lipos - r.org).normalized() * 0.5 + 0.5
            
            if h.pos.x >= 2 - EPS:
                r.color *= V(.6, 0, 1)
            if h.pos.x <= EPS - 2:
                r.color *= V(1, 0, .6)
            r.dir = rdiff

        elif h.id == 3:
            r.color *= 18
            r.kill()
        
        r.org += h.nrm * EPS * 2

        return r
