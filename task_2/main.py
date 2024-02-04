import numpy as np 
import time


start = time.time()
arr = np.random.rand(1000,1000)

print("array created in: ", time.time()-start)

# converting to bytes

bytearray = arr.tobytes()

# converting back to array

arr2 = np.frombuffer(bytearray, dtype=arr.dtype).reshape(arr.shape)

print(arr2 == arr)