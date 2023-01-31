# coding: utf-8
import numpy as np

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def erdem_correlation(array_1, array_2):
    a_12 = 0.
    a_1 = 0.
    a_2 = 0.
    for i in range(1, len(array_1)):
        a_12 += (array_1[i] - array_1[i-1]) * (array_2[i] - array_2[i-1])
        a_1 += (array_1[i] - array_1[i-1]) * (array_1[i] - array_1[i-1])
        a_2 += (array_2[i] - array_2[i-1]) * (array_2[i] - array_2[i-1])
    a_1 = np.sqrt(a_1/(len(array_1) - 1))
    a_2 = np.sqrt(a_2/(len(array_2) - 1))
    coefficient = a_12/((len(array_1)-1) * a_1 * a_2)
    return coefficient

# Code translated from the java version https://github.com/AdrienGuille/MABED/blob/b237855fb4f6ddebfb1ebf50fcbb43e1140b00e0/src/fr/ericlab/mabed/algo/Component2.java#L31
def erdem_correlation_java(ref, comp, a, b):
    array_1 = np.empty(b-a+1, dtype=np.double)
    array_2 = np.empty(b-a+1, dtype=np.double)

    for i in range(a, b+1):
        array_1[i-a] = ref[i]
        array_2[i-a] = comp[i]

    a_12 = 0.
    a_1 = 0.
    a_2 = 0.
    for i in range(1, len(array_1)):
        a_12 += (array_1[i] - array_1[i-1]) * (array_2[i] - array_2[i-1])
        a_1 += (array_1[i] - array_1[i-1]) * (array_1[i] - array_1[i-1])
        a_2 += (array_2[i] - array_2[i-1]) * (array_2[i] - array_2[i-1])

    if (len(array_1) - 1) == 0:
        a_1 = 0.
    else:
        a_1 = np.sqrt(a_1/(len(array_1) - 1))

    if (len(array_2) - 1) == 0:
        a_2 = 0.
    else:
        a_2 = np.sqrt(a_2/(len(array_2) - 1))

    if ((len(array_1)-1) * a_1 * a_2) == 0:
        coefficient = 0.
    else:
        coefficient = a_12/((len(array_1)-1) * a_1 * a_2)

    return (coefficient+1)/2


def szymkiewicz_simpson(x: set, y: set):
    """
    Szymkiewicz–Simpson coefficient
    https://en.wikipedia.org/wiki/overlap_coefficient
    """
    min_len = min(len(x), len(y))
    if min_len == 0:
        return 0.0
    return len(x.intersection(y)) / min_len

def overlap_coefficient(X: tuple, Y: tuple):
    """
    Overlap coefficient for tuples of time indices
    e.g. overlap_coefficient([3, 6], [4, 7])
    https://en.wikipedia.org/wiki/Overlap_coefficient
    """
    if X[0] > Y[1] or Y[0] > X[1]:
        # first interval ends before second begins
        # so the overlap is zero
        return 0

    intersection_cardinality = min(X[1], Y[1]) - max(X[0], Y[0])
    assert intersection_cardinality >= 0

    smallest_interval_cardinality = float(min(X[1] - X[0], Y[1] - Y[0]))

    if smallest_interval_cardinality == 0:
        return 0

    return float(intersection_cardinality / smallest_interval_cardinality)

def distance_in_time(X: tuple, Y: tuple, period: tuple = None):
    # a_beg <= a_end
    if X[0] <= Y[0]:
        a_beg, a_end, b_beg, b_end = *X, *Y
    else:
        a_beg, a_end, b_beg, b_end = *Y, *X

    if period is None:
        # compute period using input events
        period = a_beg, max(a_end, b_end)  # min(a_beg, b_beg) = a_beg

    # a_len = a_end - a_beg
    # b_len = b_end - b_beg
    period_len = period[1] - period[0]

    assert period_len > 0, "distance_in_time: period is empty"
    assert period[0] <= a_beg, "distance_in_time: implementation has a bug"
    assert period[1] >= max(a_end, b_end), "distance_in_time: implementation has a bug"

    gap = b_beg - a_end  # negative if overlap, positive if gap

    return gap / period_len


assert distance_in_time((2000, 2015), (2005, 2020)) == -0.5, "sanity check failed: events are overlapping by 50%"
assert distance_in_time((2000, 2010), (2010, 2020)) == 0, "sanity check failed: events are close to each other"
assert distance_in_time((2000, 2005), (2015, 2020)) == 0.5, "sanity check failed: events are separated by 50% of the whole period (2000-2020)"

