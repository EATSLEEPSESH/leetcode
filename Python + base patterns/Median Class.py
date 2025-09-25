class MedianFinder(object):

    def __init__(self):
        self.list = []

    def addNum(self, num):
        self.list.append(num)

    def findMedian(self):
        n = len(self.list)
        if n == 0:
            return None
        if n % 2 == 0:
            return self.list[n // 2] + self.list[(n - 1) // 2] / 2
        else:
            return self.list[n // 2]

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
