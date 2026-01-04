import sys  
import time 
import functools
import matplotlib.pyplot as plt

from PyExner import run_driver

if __name__ == "__main__":

    config_path = sys.argv[1]
    state, coords = run_driver(config_path)
    