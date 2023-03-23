import numpy as np

def argmaxes(arr):
    """Given an array, this function chooses the indexes that has the maximum value of it."""
    arg_maxes = []
    mx = arr[0]
    
    for i in range(len(arr)):
        if arr[i] > mx:
            mx = arr[i]
            arg_maxes = [i]
        elif arr[i] == mx:
            arg_maxes.append(i)
            
    return arg_maxes

def exp_moving_avg(arr, beta=0.9):
    """Given a time-series, calculate an exponentially moving average of it."""
    n = arr.shape[0]
    mov_avg = np.zeros(n)
    mov_avg[0] = (1-beta) * arr[0]
    for i in range(1, n):
        mov_avg[i] = beta * mov_avg[i-1] + (1-beta) * arr[i]
    return mov_avg