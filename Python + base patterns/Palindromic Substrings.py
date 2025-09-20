class Solution(object):
    def countSubstrings(self, s):
        answer = 0
        def expand(l,r):
            cnt = 0
            while l >= 0 and r < len(s) and s[l] == s[r]:
                cnt += 1
                l -= 1
                r += 1
            return cnt

        for i in range(len(s)):
            answer += expand(i, i)
            answer += expand(i, i + 1)
        return answer

