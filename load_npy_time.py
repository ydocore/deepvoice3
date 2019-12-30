import numpy as np
import time
from os.path import dirname, join

def load_npy(path):
    meta = join(path,"train.txt")
    with open(meta,"rb") as f:
        lines = f.readlines()
    #l = lines[0].decode("utf-8").split("|")
    frame_lengths = list(map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))
    arg_max_len = np.argmax(frame_lengths)

    l = lines[arg_max_len].decode("utf-8").split("|")
    #return l
    x = []
    for t,i in enumerate(l):
        if t != 2 and t != 6 and t != 7:
            x.append(np.load(path+i))
    return x

def time_count(path):
    start = time.time()
    load_npy(path)
    end = time.time()-start
    return end
