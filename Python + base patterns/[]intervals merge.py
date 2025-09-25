#def IntervalsMerge(array):
#    arr = sorted(array)
#    indx = 1
#    while indx <= len(arr) - 1:
#        if arr[indx][0] <= arr[indx - 1][1]:
#            new_int = [arr[indx - 1][0], max(arr[indx - 1][1], arr[indx][1])]
#            arr.remove(arr[indx])
#            arr.remove(arr[indx - 1])
#            arr.insert(indx - 1, new_int)
#        else:
#            indx += 1
#    return arr
#
#print(IntervalsMerge([]))
def mergeIntervals(intervals):
    if not intervals:
        return []

    # Сортируем по левым границам
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_end = merged[-1][1]

        if start <= last_end:  # пересекаются
            merged[-1][1] = max(last_end, end)
        else:  # новый интервал
            merged.append([start, end])

    return merged
