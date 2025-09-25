class Solution(object):
    def isValid(self, s):
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        opens = set(pairs.values())

        for ch in s:
            if ch in opens:
                stack.append(ch)
            else:
                if not stack or stack[-1] != pairs.get(ch):
                    return False
                stack.pop()
        return not stack
