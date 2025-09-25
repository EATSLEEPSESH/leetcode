class Solution(object):
    def maxProduct(self, nums):
        curr_max, curr_min, answer = nums[0]
        for i in nums[1:]:
            if i < 0:
                curr_max, curr_min = curr_min, curr_max
            curr_max = max(i, curr_max * i)
            curr_min = min(i, curr_min * i)
            answer = max(answer, curr_max)

        return answer


print(Solution().maxProduct([2, 3, -2, 4]))
