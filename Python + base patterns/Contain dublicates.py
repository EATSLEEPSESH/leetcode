class Solution(object):

    def containsDuplicate(self, nums):
        values = set()
        for el in nums:
            if el in values:
                return True
            else:
                values.add(el)
        return False

    class Solution(object):
        def containsDuplicate(self, nums):
            freq = {}
            for x in nums:
                freq[x] = freq.get(x, 0) + 1

            # any вернёт True, если хотя бы одно значение >= 2
            return any(count >= 2 for count in freq.values())