class Solution(object):
    def threeSum(self, nums):
        nums.sort()
        result = []
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            l, r = i + 1, n - 1

            while l < r:
                sums = nums[i] + nums[l] + nums[r]
                if sums < 0:
                    l += 1
                elif sums > 0:
                    r -= 1
                else:
                    result.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while r > l and nums[r] == nums[r + 1]:
                        r -= 1
        return result


print(Solution().threeSum([1,1,1]))

