#!/usr/bin/env python3

def main():
    import numpy as np
    import makePoints
    
    rays_sp = np.zeros((5,4,3))
    rays_slv =np.random.randint(low=0,high=100,size=(5,4,3))
    
    test = np.zeros(rays_slv.shape)
    for k1 in range(5):
        for k2 in range(4):
            test[k1,k2,:] = rays_slv[k1,k2,:].copy()/np.sqrt(np.sum(np.square(rays_slv[k1,k2,:].copy())))
    
    rays_slv = test.copy()
    M = 15000
    stepSize = 15.0
    
    test_out = makePoints.makePoints3D(M,stepSize, rays_sp, rays_slv)
    
    print('Successfully completed')


if __name__=='__main__':
   main()
