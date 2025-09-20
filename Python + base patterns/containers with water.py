class Solution(object):
    def maxArea(self, height):
        l, r = 0, len(height) - 1
        best = 0
        while l < r:
            best = max(best, (r-l) * min(height[r], height[l]))
            if height[r] < height[l]:
                r -= 1
            else:
                l += 1
        return best


print(Solution().maxArea([1,8,6,2,5,4,8,3,7]))
