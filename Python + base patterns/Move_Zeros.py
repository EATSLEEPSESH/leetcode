class Solution(object):

    def moveZeroes(self, nums):
        insert_pos = 0
        for num in nums:
            if num != 0:
                nums[insert_pos] = num
                insert_pos += 1
        while insert_pos < len(nums):
            nums[insert_pos] = 0
            insert_pos += 1
        return nums


print(Solution().moveZeroes([1,0,2,0,3]))