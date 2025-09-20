from array import array

def solve():
    n_str, k_str = input().split()
    n = int(n_str)
    k = int(k_str)
    cats = []
    while len(cats) < n:
        cats.extend(map(int, input().split()))
    if k == 0:
        print(0)
        return
    M = 1_000_001
    breed_count = array('I', [0]) * M
    distinct = 0
    l = 0
    best = 0
    for r in range(n):
        x = cats[r]
        if breed_count[x] == 0:
            distinct += 1
        breed_count[x] += 1
        while distinct > k:
            y = cats[l]
            breed_count[y] -= 1
            if breed_count[y] == 0:
                distinct -= 1
            l += 1
        length = r - l + 1
        if length > best:
            best = length
    print(best)

if __name__ == "__main__":
    solve()
