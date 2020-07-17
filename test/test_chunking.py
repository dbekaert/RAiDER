import numpy as np
import pytest

from RAiDER.delayFcns import chunk,makeChunksFromInds,makeChunkStartInds

# The purpose of these tests is to verify the chunking done on the query points during is
# done correctly for various dimensions

def checkChunks(chunk1, chunk2):
    check = []
    for k1,L in enumerate(chunk2):
        for k2,arr in enumerate(L):
            if np.allclose(arr, chunk1[k1][k2]):
                check.append(True)
            else:
                check.append(False)
    return all(check)

def test_makeChunkStartInds1D():
    chunkStartInds = [(0,), (1,), (2,), (3,), (4,)]
    assert chunkStartInds == makeChunkStartInds((1,),(5,))

def test_makeChunkStartInds2D():
    chunkStartInds = [(0, 0), (0, 1), (3, 0), (3, 1), (6, 0), (6, 1)]
    assert chunkStartInds == makeChunkStartInds((3,1), (9,2))

def test_makeChunkStartInds3D():
    chunkStartInds = (
        [(0, 0, 0),
	 (0, 2, 0),
	 (0, 4, 0),
	 (0, 6, 0),
	 (2, 0, 0),
	 (2, 2, 0),
	 (2, 4, 0),
	 (2, 6, 0)] 
	)
    assert chunkStartInds == makeChunkStartInds((2,2,16), (4,8,16))

def test_makeChunksFromInds1D():
    in_shape = (4,)
    chunkSize = (2,)
    ind = [(0,), (2,)]
    chunks = [[np.array([0, 1])], [np.array([2, 3])]]
    assert checkChunks(chunks, makeChunksFromInds(ind,chunkSize,in_shape)) 

def test_makeChunksFromInds2D():
    in_shape = (4,4)
    chunkSize = (2,2)
    ind = [(0, 0), (0, 2), (2, 0), (2, 2)]
    chunks = [[np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
              [np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])],
              [np.array([2, 2, 3, 3]), np.array([0, 1, 0, 1])],
              [np.array([2, 2, 3, 3]), np.array([2, 3, 2, 3])]]
    assert checkChunks(chunks, makeChunksFromInds(ind,chunkSize,in_shape)) 

def test_makeChunksFromInds3D():
    in_shape = (2,2,2)
    chunkSize = (1,1,2)
    ind = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    chunks = [[np.array([0, 0]), np.array([0, 0]), np.array([0, 1])],
              [np.array([0, 0]), np.array([1, 1]), np.array([0, 1])],
              [np.array([1, 1]), np.array([0, 0]), np.array([0, 1])],
              [np.array([1, 1]), np.array([1, 1]), np.array([0, 1])]]
    assert checkChunks(chunks, makeChunksFromInds(ind,chunkSize,in_shape)) 

def test_chunk1D():
    chunkSize = (10,)
    in_shape = (50,)
    chunks = [[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
              [np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])],
              [np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])],
              [np.array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39])],
              [np.array([40, 41, 42, 43, 44, 45, 46, 47, 48, 49])]]
    assert checkChunks(chunks, chunk(chunkSize,in_shape)) 

def test_chunk2D():
    chunkSize = (2,2)
    in_shape = (4,4)
    chunks = [[np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
              [np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])],
              [np.array([2, 2, 3, 3]), np.array([0, 1, 0, 1])],
              [np.array([2, 2, 3, 3]), np.array([2, 3, 2, 3])]]
    assert checkChunks(chunks, chunk(chunkSize,in_shape)) 

def test_chunk3D():
    chunkSize = (2,2,2)
    in_shape = (2,4,4)
    chunks = [[np.array([0, 0, 0, 0, 1, 1, 1, 1]),
               np.array([0, 0, 1, 1, 0, 0, 1, 1]),
               np.array([0, 1, 0, 1, 0, 1, 0, 1])],
              [np.array([0, 0, 0, 0, 1, 1, 1, 1]),
               np.array([0, 0, 1, 1, 0, 0, 1, 1]),
               np.array([2, 3, 2, 3, 2, 3, 2, 3])],
              [np.array([0, 0, 0, 0, 1, 1, 1, 1]),
               np.array([2, 2, 3, 3, 2, 2, 3, 3]),
               np.array([0, 1, 0, 1, 0, 1, 0, 1])],
              [np.array([0, 0, 0, 0, 1, 1, 1, 1]),
               np.array([2, 2, 3, 3, 2, 2, 3, 3]),
               np.array([2, 3, 2, 3, 2, 3, 2, 3])]]
    assert checkChunks(chunks, chunk(chunkSize,in_shape)) 

