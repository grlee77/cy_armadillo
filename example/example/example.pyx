import numpy as np
cimport numpy as np

include "cyarma.pyx"

def svd_wrapper(np.ndarray[np.double_t, ndim=2] A): #, char * method='dc'):
    
    m=A.shape[0]
    n=A.shape[1]

    cdef:
        mat Amat = numpy_to_mat(A)
        mat U = mat(m,m)
        vec s = vec(min(m,n))
        mat V = mat(n,n)
        bool status
        cdef np.ndarray[np.double_t,ndim=2] Uout
        cdef np.ndarray[np.double_t,ndim=1] Sout
        cdef np.ndarray[np.double_t,ndim=2] Vout    

    status = svd( U, s, V, Amat) #, 'std')
    
    if not status:
        raise Exception("SVD failed to converge")
    else:
        Uout = mat_to_numpy(U,None)
        Sout = vec_to_numpy(s,None)
        Vout = mat_to_numpy(V,None)
        return ( Uout, Sout, Vout)

def example(np.ndarray[np.double_t, ndim=2] X):

    cdef mat aX = numpy_to_mat(X)

    cdef mat XX = aX.t() * aX
    cdef mat ch = chol(XX)
    ch.raw_print()
    print np.linalg.cholesky(np.dot(X.T,X))
    cdef np.ndarray[np.double_t,ndim=2] Y = mat_to_numpy(ch, None)

    
    cdef double ctest[10][10]
    cdef double [:,:] test = ctest
    cdef int i, j
    for i in range(10):
        for j in range(10):
            test[i,j] = i+j
    cdef mat *test2 = new mat(<double*> ctest, 10, 10, False, True)
    test2.raw_print()
    
    
    #singular values
    cdef vec s = svd(XX)
    s.raw_print()
    print(np.linalg.svd(np.dot(X.T,X),compute_uv=False))

    U,S,V=svd_wrapper(np.random.random((100,50)))
    print(S)

    return Y
