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


def overlap_coefficient(interval_0, interval_1):
    intersection_cardinality = float(min(interval_0[1], interval_1[1]) - max(interval_0[0], interval_1[0]))
    smallest_interval_cardinality = float(min(interval_0[1] - interval_0[0], interval_1[1] - interval_1[0]))
    
    if(smallest_interval_cardinality==0):
        return 0
    
    return float(intersection_cardinality / smallest_interval_cardinality)
