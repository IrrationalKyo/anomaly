import numpy as np
import math
from sklearn import mixture

''' memory_trace is a list of memories accesses in sequential order '''
def creat_heatmap(memory_trace, addr_base, size, delta):

    heatmap = np.zeros(size)
    for addr in memory_trace:
        offset = addr - addr_base
        index = math.floor(offset/delta)
        if index >= size:
            break
        heatmap[index] += 1
    return heatmap

def learn_heatmap():

