import os
import pickle
import sys
import matplotlib.pyplot as plt

def analysis(stat_file):
    f = open(stat_file, 'r')
    line = f.readlines()
    for l in line:
        l = eval(l)
        import IPython; IPython.embed()

if __name__ == "__main__":
    stat_file = sys.argv[1]
    analysis(stat_file)


