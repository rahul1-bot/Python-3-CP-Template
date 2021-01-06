#-- Competitive Programming Python 3.9 --#
#-- Ultra-Fast Inputs and Outputs --#
import atexit, io, sys, numba                                    # System Libraries
import collections, itertools, functools, operator               # Data Structures
import bisect, heapq, re, math, statistics                       # Algorithms 
from typing import Any, Dict, List, Tuple, Callable, Iterator    # Python 3.9 Static Typing Class    

#</buffer-io>
buffer = io.BytesIO()
sys.stdout = buffer

@atexit.register
def write() -> None:
    sys.__stdout__.write(buffer.getvalue().decode("utf-8"))
    
#</buffer-io>
sys.stdin = open("Input.txt", "r")
sys.stdout = open("Output.txt", "w")
sys.setrecursionlimit(100000000)

# </Macros: Inputs>
intp = lambda: int(input())
strp = lambda: input().strip()
jn   = lambda x,l: x.join(map(str, l))
strl = lambda: list(input().strip())
mul  = lambda: map(int, input().strip().split())
mulf = lambda: map(float, input().strip().split())
seq  = lambda: list(map(int, input().strip().split()))

@numba.jit                                                      # Speed equivalent to C
def factorial(x: int) -> int:       
    if not x >=1:
        return 1
    return x * factorial(x-1)

# Driver Code
if __name__ == "__main__":
    print(factorial(4))
    mapper: Dict[Any, Any] = {
        int(index): item ** 4 
        for index, item in enumerate(range(20)) 
        if index % 2
    }
    print(mapper)
    print("Hello")