from collections import Counter
def KMostPopular(array, k):
    answer = Counter(array)
    return [num for num, _ in answer.most_common(k)]


print(KMostPopular([1, 1, 2, 2, 2], 1))
