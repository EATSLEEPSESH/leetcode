class Solution(object):
    def lengthOfLongestSubstring(self, s):
        seen = set()
        best = 0
        l = 0
        for r, ch in enumerate(s):
            while ch in seen:
                seen.remove(s[l])
                l+=1
            seen.add(ch)
            best = max(best, r - l + 1)
        return best



print(Solution().lengthOfLongestSubstring("pwwkew"))
