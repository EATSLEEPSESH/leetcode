from collections import deque

def num_islands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0

    for r in range(m):
        for c in range(n):
            if grid[r][c] == '1':
                count += 1
                q = deque([(r, c)])
                grid[r][c] = '0'  # помечаем как посещённую

                while q:
                    i, j = q.popleft()
                    for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                            grid[ni][nj] = '0'
                            q.append((ni, nj))
    return count
