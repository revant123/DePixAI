# blocky_score.pyx
cimport numpy as np
import numpy as np

def calculate_blocky_score(np.ndarray[np.uint8_t, ndim=2] edges, int block_size):
    cdef int height = edges.shape[0]
    cdef int width = edges.shape[1]
    cdef double blocky_score = 0.0

    cdef int y, x
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = edges[y:y+block_size, x:x+block_size]
            variance = np.var(block)
            edge_density = np.sum(block) / (block_size * block_size)
            blocky_score += variance * edge_density

    return blocky_score
