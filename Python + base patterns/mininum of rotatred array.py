class Solution(object):
    def findMin(self, nums):
        n = len(nums)
        l, r = 0, n - 1
        if n == 0:
            return []
        elif n < 3:
            return min(nums)
        while l < r:
            if nums[l] > nums[r]:
                l += 1
                r -= 1
            else:
                break
        return min(nums[l + 1], nums[l])


print(Solution().findMin([11,13,15,17]))
print(Solution().findMin([3,1,2]))