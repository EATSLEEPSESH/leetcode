class Solution(object):
    def search(self, nums, target):
        n = len(nums)
        l, r = 0, n - 1
        while l < r:
            mid = (l + r) // 2
            if nums[l] < nums[r]:
                if target < nums[mid]:
                    r = mid - 1
                elif nums[mid] == target:
                    return mid
                else:
                    l = mid + 1
            else:
                if target < nums[mid]:
                    l = mid + 1
                elif nums[mid] == target:
                    return mid
                else:
                    r = mid - 1
        return l if nums[l] == target else -1


print(Solution().search([3,5,1], 3))


