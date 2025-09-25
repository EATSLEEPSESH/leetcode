from typing import List

def rob(nums: List[int]) -> int:
    take = 0  # максимум, если берем текущий
    skip = 0  # максимум, если пропускаем текущий
    for x in nums:
        new_take = skip + x
        new_skip = max(skip, take)
        take, skip = new_take, new_skip
    return max(take, skip)