# Tests for array conversion functions
import pytest
import numpy as np
from arrary_converter import convert_array_to_bytes, convert_bytes_to_array


arr_shape = (1000, 1000)
arr_dtype = np.float32

@pytest.fixture
def random_array():
    return np.random.rand(*arr_shape).astype(arr_dtype)

def test_array_to_bytes_conversion(random_array):
    byte_array = convert_array_to_bytes(random_array)
    assert isinstance(byte_array, bytes)

def test_bytes_to_array_conversion(random_array):
    byte_array = convert_array_to_bytes(random_array)
    array_from_bytes = convert_bytes_to_array(byte_array, arr_shape, arr_dtype)
    assert np.array_equal(array_from_bytes, random_array)

if __name__ == "__main__":
    pytest.main()