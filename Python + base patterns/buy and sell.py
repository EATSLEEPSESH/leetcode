class Solution(object):
    def maxProfit(self, prices):
        profit = 0
        min_price = 10000000000
        for price in prices:
            min_price = min(min_price, price)
            profit = max(price - min_price, profit)
        return profit