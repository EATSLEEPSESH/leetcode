class Solution(object):
    def longestPalindrome(self, s):
        if not s:
            return ""

        def expand(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return l + 1, r

        start, end = 0, 1
        for i in range(len(s)):
            l1, r1 = expand(i, i)
            l2, r2 = expand(i, i + 1)
            if end - start < r1 - l1:
                start, end = l1, r1
            if end - start < r2 - l2:
                start, end = l2, r2
        return s[start:end]
