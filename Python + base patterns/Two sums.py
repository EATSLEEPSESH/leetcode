class Solution(object):

    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            dlc = target - num
            if dlc in seen:
                return [seen[dlc], i]
            seen[num] = i


s = Solution()
print(s.twoSum([1, 2, 7, 5], 9))
