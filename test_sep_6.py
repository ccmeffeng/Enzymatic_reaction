import numpy as np

def rev(fname):
    n = np.load('%s.npy'%fname).tolist()
    for i in range(len(n)):
        n[i] = n[i][::-1]
    n = np.array(n)
    np.save('%s_r.npy'%fname, n)

rev('test_set')
rev('training_set')
rev('validation_set')
