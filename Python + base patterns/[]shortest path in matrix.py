from collections import deque

def shortest_path_in_matrix(grid):
    n = len(grid)
    if n == 0:
        return -1
    if len(grid[0]) != n:  # если вдруг не квадрат — можно убрать, если гарантирован квадрат
        return -1
    # Старт/финиш заблокирован
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    # 4-направления: вниз, вправо, вверх, влево
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]
    visited = [[False]*n for _ in range(n)]
    q = deque()
    # В очередь кладём (row, col, dist), где dist — длина пути в шагах от старта
    q.append((0, 0, 0))
    visited[0][0] = True

    while q:
        r, c, d = q.popleft()
        if r == n-1 and c == n-1:
            return d  # возвращаем количество шагов (как в твоём примере: 4)

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc, d + 1))

    return -1
