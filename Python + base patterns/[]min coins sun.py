def coinChange(coins, amount):
    INF = amount + 1                    # «бесконечность»
    dp = [INF] * (amount + 1)
    dp[0] = 0                           # 0 монет, чтобы набрать сумму 0

    for c in coins:
        for x in range(c, amount + 1):  # unbounded: монетами можно пользоваться многократно
            dp[x] = min(dp[x], dp[x - c] + 1)

    return dp[amount] if dp[amount] != INF else -1