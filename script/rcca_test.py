import numpy as np
from scipy import linalg as SLA
import matplotlib.pyplot as plt

def fileinput():
    datas = []    
    for line in open('rcca_testdata.txt', 'r'):
        data = []
        for d in line[:-1].split('\t'):
            data.append(float(d))
        datas.append(data)
    #print datas    
    return datas

def rcca(X, Y, reg):   
    X = np.array(X)
    Y = np.array(Y)
    n, p = X.shape
    n, q = Y.shape
        
    # zero mean
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
        
    # covariances
    S = np.cov(X.T, Y.T, bias=1)
    
    SXX = S[:p,:p]
    SYY = S[p:,p:]
    SXY = S[:p,p:]

    ux, lx, vxt = SLA.svd(SXX)
    #uy, ly, vyt = SLA.svd(SYY)
    
    #print lx
    #正則化
    Rg = np.diag(np.ones(p)*reg)
    SXX = SXX + Rg
    SYY = SYY + Rg

    sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
    sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
    M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
    A, r, Bh = SLA.svd(M, full_matrices=False)
    B = Bh.T      
    #r = self.reg*r
    return r[0], lx[0], lx[0] 
    
    
if __name__ == "__main__":
    
    X, Y = fileinput(), fileinput()
    
    rs = []
    for i in range(1000):
        reg = 0.00001 * 10 * i
        r, lx, ly = rcca(X, Y, reg)
        rs.append(r)
        print reg,":",r,",",lx,",",ly
    
    plt.plot(rs)