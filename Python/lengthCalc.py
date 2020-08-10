from statistics import median, mean

def lengthCalc(seqList):
    """calculates length stats

    Keyword arguments:
    seqList: a list of squence
    """
    lenList = map(len, seqList)
    maxLen = max(lenList)
    minLen = min(lenList)
    meanLen = mean(lenList)
    medLen = median(lenList)

    return maxLen, minLen, meanLen, medLen
