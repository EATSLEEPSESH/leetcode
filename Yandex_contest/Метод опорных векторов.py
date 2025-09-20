import math
import sys

TAU = 2.0 * math.pi


def angle(x, y):
    a = math.atan2(y, x)
    if a < 0.0:
        a += TAU
    return a


def feasible_r(angles, norms, r):
    n = len(angles)
    if n == 0:
        return False
    L = angles[0] - math.acos(min(1.0, r / norms[0]))
    R = angles[0] + math.acos(min(1.0, r / norms[0]))
    mid = 0.5 * (L + R)
    for i in range(1, n):
        if r > norms[i] + 1e-15:
            return False
        wi = math.acos(min(1.0, r / norms[i]))
        ci = angles[i]
        k = round((mid - ci) / TAU)
        Li = ci - wi + k * TAU
        Ri = ci + wi + k * TAU
        if Ri < L:
            Li += TAU
            Ri += TAU
        elif Li > R:
            Li -= TAU
            Ri -= TAU
        if Li > R or Ri < L:
            return False
        if Li > L:
            L = Li
        if Ri < R:
            R = Ri
        mid = 0.5 * (L + R)
    return True


def solve():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n1 = int(next(it))
    A = []
    for _ in range(n1):
        x = float(next(it))
        y = float(next(it))
        A.append((x, y))
    n2 = int(next(it))
    B = []
    for _ in range(n2):
        x = float(next(it))
        y = float(next(it))
        B.append((x, y))
    x0 = float(next(it))
    y0 = float(next(it))

    for x, y in A + B:
        if abs(x - x0) < 1e-12 and abs(y - y0) < 1e-12:
            print(-1)
            return

    v_angles = []
    v_norms = []
    for x, y in A:
        dx = x - x0
        dy = y - y0
        nrm = math.hypot(dx, dy)
        v_angles.append(angle(dx, dy))
        v_norms.append(nrm)

    w_angles = []
    w_norms = []
    for x, y in B:
        dx = x - x0
        dy = y - y0
        nrm = math.hypot(dx, dy)
        w_angles.append(angle(dx, dy))
        w_norms.append(nrm)

    angles1 = v_angles + [(a + math.pi) % TAU for a in w_angles]
    norms1 = v_norms + w_norms

    angles2 = [(a + math.pi) % TAU for a in v_angles] + w_angles
    norms2 = v_norms + w_norms

    ok1 = feasible_r(angles1, norms1, 0.0)
    ok2 = feasible_r(angles2, norms2, 0.0)
    if not ok1 and not ok2:
        print(-1)
        return

    def max_r(angles, norms):
        lo = 0.0
        hi = min(norms)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if feasible_r(angles, norms, mid):
                lo = mid
            else:
                hi = mid
        return lo

    ans = 0.0
    if ok1:
        ans = max(ans, max_r(angles1, norms1))
    if ok2:
        ans = max(ans, max_r(angles2, norms2))

    print("{:.10f}".format(ans))


if __name__ == "__main__":
    solve()
