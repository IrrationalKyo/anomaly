import numpy as np
import math
from sklearn import mixture
import matplotlib.pyplot as plt

'''
    memory_trace is a list of memories accesses in sequential order
    bin: size of a slot
    bin_count: number of slots
'''

def file_to_list(file_name):
    content = []
    with open(file_name,"r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]  
    content = [x.split("~") for x in content]  
    content = [(int(x[0],16),int(x[1],16)) for x in content]  

    return content


def create_heatmap(memory_trace, addr_base, bin_size, bin_count):
    
    heatmap = np.zeros((bin_count))

    for access_interval in memory_trace:
        addr_begin = int(access_interval[0]) - addr_base
        addr_end = int(access_interval[1]) - addr_base

        index_begin = math.floor(addr_begin/bin_size)
        index_end = math.floor(addr_end/bin_size)

        if index_end >= bin_count or index_begin >= bin_count:
            continue
        if index_begin == index_end:
            heatmap[index_begin] += 1
        else:
            index_diff = index_end - index_begin
            for i in range(index_diff):
                heatmap[index_begin + i] += float(1/(index_diff+1))
    return heatmap

if __name__ == "__main__":
    ftl = file_to_list("../../sim/gem5/Log/bitcnts_x86.memtrace")
    min_addr = 99999999
    max_addr = 0

    for ai in ftl:
        if ai[0] < min_addr:
            min_addr = ai[0]
        if ai[1] > max_addr:
            max_addr = ai[1]

    print("min_addr: " + str(min_addr))
    print("max_addr: " + str(max_addr))
    mhm = create_heatmap(ftl, 0, 512,2000)
    np_mhm = np.asarray(mhm).reshape(-1,50)

    fig, ax = plt.subplots()

    ax.imshow(np_mhm, cmap=plt.cm.Blues, interpolation='nearest')
    heatmap = ax.pcolor(np_mhm, cmap=plt.cm.Blues)
    plt.colorbar(heatmap)
    plt.show()



    
