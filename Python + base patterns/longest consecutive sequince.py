class Solution(object):
    def longestConsecutive(self, nums):
        a = set(nums)
        l = 0
        answer = 0
        for el in a:
            if el - 1 not in a:
                while el + 1 in a:
                    l += 1
                    el += 1
                answer = max(l, answer)
        return answer




print(Solution().longestConsecutive([100,4,200,1,3,2]))
