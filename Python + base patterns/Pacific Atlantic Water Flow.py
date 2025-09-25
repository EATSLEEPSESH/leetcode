class Solution(object):
    def pacificAtlantic(self, heights):
        if not heights or not heights[0]:
            return []
        m, n = len(heights), len(heights[0])
        Pasific = [[False] * n for i in range(m)]
        Atlantic = [[False] * n for i in range(m)]

        def dfs(row, col, visited, prev_hei):
            if (row < 0 or row >= m or col < 0 or col >= n or visited[row][col] or
                    heights[row][col] < prev_hei):
                return
            visited[row][col] = True
            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                dfs(row + dr, col + dc, visited, heights[row][col])

        for col in range(n):
            dfs(0, col, Pasific, heights[0][col])
        for row in range(m):
            dfs(row, 0, Pasific, heights[row][0])

        for col in range(n):
            dfs(m - 1, col, Atlantic, heights[m - 1][col])
        for row in range(m):
            dfs(row, n - 1, Atlantic, heights[row][n - 1])

        res = []
        for i in range(m):
            for j in range(n):
                if Pasific[i][j] and Atlantic[i][j]:
                    res.append([i, j])
        return res


print(Solution().pacificAtlantic([[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]))
