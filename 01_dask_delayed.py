# Parallelize code with dask.delayed
# Code Credit: https://github.com/dask/dask-tutorial/

# 1- Make toy functions, inc and add, that sleep for a while to simulate work
from time import sleep

def inc(x):
	sleep(1)
	return x + 1

def add(x, y):
	sleep(1)
	return x + y 

# 3 seconds to run each function sequentially, one after the other 
time.sleep(3) 
x = inc(1)
y = inc(2)
z = add(x, y)
