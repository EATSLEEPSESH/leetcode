from collections import Counter

class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if k == 0:
            return 0
        counter = Counter()
        l = 0
        best = 0

        for r, ch in enumerate(s):
            counter[ch] += 1

            # Сжимаем окно, если символов больше k
            while len(counter) > k:
                counter[s[l]] -= 1
                if counter[s[l]] == 0:
                    del counter[s[l]]
                l += 1

            best = max(best, r - l + 1)

        return best