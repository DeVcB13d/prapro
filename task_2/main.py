import numpy as np 
import time

arr_shape = [1000,1000]
arr_dtype = np.float64


start = time.time()
arr1 = np.random.rand(*arr_shape).astype(arr_dtype)


print("array created in: ", time.time()-start)

# converting to bytes
bytearray = arr1.tobytes()


# converting back to array
arr2 = np.frombuffer(bytearray, dtype=arr_dtype).reshape(arr_shape)

print(arr1 == arr2)

