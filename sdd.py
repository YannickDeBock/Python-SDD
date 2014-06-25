# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:58:36 2014

@author: Yannick De Bock

 Python implementation of the Semi Discrete Decomposition
 Copyright (c) 2014 Yannick De Bock 

 This program is derived from: 
 SDDPACK: Software for the Semidiscrete Decomposition.
 Copyright (c) 1999 Tamara G. Kolda and Dianne P. O'Leary.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 Contact info: email: yannick.debock@kuleuven.be
          paper mail: Celestijnenlaan 300 A Bus 2422, B-3001 Heverlee (Leuven), Belgium
"""

import numpy as np, math, operator

################# Function to compute SDD ###########################
def sdd(A,kmax=100,alphamin=0.01,lmax=100,rhomin=0,yinit=1):

#    SDD  Semidiscrete Decomposition.
#
#   [D, X, Y, _, _, _] = SDD(A) produces discrete matrices X and Y and a vector D
#   such that X * diag(D) * Y.T is the 100-term SDD that approximates A.
#
#   [D, X, Y, ITS, _, _] = SDD(...) also returns the number of inner iterations
#   for each outer iteration.  
#
#   [D, X, Y, ITS, RHO, _] = SDD(...) also returns a vector RHO containing the
#   norm-squared of the residual after each outer iteration.
#
#   [D, X, Y, ITS, RHO, IITS] = SDD(...) also returns a vector IITS
#   containing the number of extra matrix-vector multiplies used in the
#   initialization of each inner iteration when using C=1 (see below). 
#
#   [...] = SDD(A, K) produces a K-term SDD. The default is 100.
#
#   [...] = SDD(A, K, TOL) stops the inner iterations after the
#   improvement is less than TOL. The default is 0.01.
#
#   [...] = SDD(A, K, TOL, L) specifies that the maximum number of inner
#   iterations is L. The default is 100.
#
#   [...] = SDD(A, K, TOL, L, R) produces an SDD approximation that
#   either has K terms or such that the frobenius norm of (A - X * diag(D) * Y.') < R. Where ' means transpose and diag constructs a diagonal matrix
#   The default is zero.
#
#   [...] = SDD(A, K, TOL, L, R, C) sets the choice for initializing y in
#   the inner iterations as follows: 
#
#      C = 1       Threshold
#      C = 2       Cycling
#      C = 3,      All elements of y are set to 1.
#      C = 4,      Every 100th element of y is set to 1 starting with 1.
#
#   Default is C = 1.
#
#
#   Reconstruction of the data can be obtained by np.dot(np.dot(X,np.diag(D.flatten().tolist())),Y.T)

### Check Input Arguments    
    
    try: 
        'A'
    except NameError:
        print 'Incorrect number of inputs.'
    
    if 'rhomin' in locals():
        rhomin = math.pow(rhomin,2)
    idx = 0             # only used for yinit = 1 (python is zero-based contrary to matlab)
    
    # Initialization    
    
    [m,n] = A.shape         # size of A
    rho = math.pow(np.linalg.norm(A,'fro'),2)   # squared residual norm
    
    iitssav = np.zeros((kmax))
    xsav = np.zeros((m,kmax))
    xsav = np.asarray(xsav)
    ysav = np.zeros((n,kmax))
    ysav = np.asarray(ysav)
    dsav = np.zeros((kmax,1))
    itssav = np.zeros((kmax))    
    rhosav = np.zeros((kmax))
    A = np.asarray(A)
    betabar = 0    
    
    # Outer loop
    
    for k in range(0,kmax):
        
        # Initialize y for inner loop
        
        if yinit == 1:          # Threshold
            s = np.zeros((m,1))
            iits = 0
            while math.pow(np.linalg.norm(s),2) < (float(rho)/n):                
                y = np.zeros((n,1))                     
                y[idx] = 1
                s = np.dot(A,y)
                if k>0:       # python is zero-based             
                    s = s - (np.dot(xsav,(np.multiply(dsav,(np.dot(ysav.T,y))))))                    
                    
                idx = np.mod(idx, n) + 1
                if idx == n:        # When idx reaches n it should be changed to zero (otherwise an index out of bounds error will occur)
                    idx = 0
                iits = iits + 1
            iitssav[k] = iits
        elif yinit == 2:        # Cycling Periodic Ones
           y = np.zeros((n,1))
            index = np.mod(k-1,n)+1
            if index < n:                 
                y[index] = 1
            else:
                y[0] = 1   
        elif yinit == 3:        # All Ones
            y = np.ones((n,1))
        elif yinit == 4:        # Periodic Ones
            y = np.zeros((n,1))
            ii = np.arange(0,n,100)
            for i in ii: # python is zero-based
                y[i] = 1 
        else:
            try:
                pass
            except ValueError:
                print 'Invalid choice for C.'
                
        # Inner loop
        
        for l in range (0,lmax):
            
            # Fix y and Solve for x
            
            s = np.dot(A,y)            
            if k > 0:       # python is zero-based
                s = s - (np.dot(xsav,(np.multiply(dsav,(np.dot(ysav.T,y))))))
            
            [x, xcnt, _] = sddsolve(s, m)

            # Fix x and Solve for y
            
            s = np.dot(A.T,x)
            if k > 0:
                s = s - (np.dot(ysav,(np.multiply(dsav,(np.dot(xsav.T,x))))))
            
            [y, ycnt, fmax] = sddsolve(s, n)
            
            # Check Progress
            
            d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)

            beta = math.pow(d,2) * ycnt * xcnt
            
            if l > 0: # python is zero-based
                alpha = (beta - betabar) / betabar
                if alpha <= alphamin:
                    break
            
            betabar = beta
        
        # Save
        
        xsav[:, k] = x.T            # shape conflict (matlab deals with this internally)        
        ysav[:, k] = y.T
        dsav[k, 0] = d              # python is zero-based        
        rho = max([rho-beta,0])
        rhosav[k] = rho
        itssav[k] = l
        
        # Threshold Test
        
        if rho <= rhomin:
            break
        
    return dsav, xsav, ysav, itssav, rhosav, iitssav

################# SDD subproblem solver ############################
def sddsolve(s, m):
#   SDDSOLVE Solve SDD subproblem
#
#   [X, _] = SDDSOLVE(S, M) computes max (X' * S) / (X' * X) where M is the
#   size of S and ' means transpose.  
#
#   [X, I, _] = SDDSOLVE(S, M) additionally returns number of nonzeros in X.
#
#   [X, I, F] = SDDSOLVE(S, M) additionally returns value of function at the
#   optimum.  
#
#For use with SDD.
#Yannick De Bock, KU Leuven, 2014
#
#Derived from SDDPACK 
#Tamara G. Kolda, Oak Ridge National Laboratory, 1999.
#Dianne P. O'Leary, University of Maryland and ETH, 1999.

    x = np.zeros((m,1))   
    x = np.asarray(x)
    
    for i in range(0,m):        # python is zero-based
        if s[i] < 0:
            x[i,0] = -1         # python is zero-based
            s[i] = -s[i]
        else:
            x[i,0] = 1          # python is zero-based
    
    sorted_array =sorted(enumerate(-s), key=operator.itemgetter(1)) # Sort array and get index of original unsorted data
    sorted_array = np.asarray(sorted_array)
    sorts = -sorted_array[:,1]
    indexsort = sorted_array[:,0]
    
    f = np.zeros((m))
    f = np.asfarray(f)
    f[0] = sorts[0]             # python is zero-based
    for i in range(1,m):
        f[i] = sorts[i] + f[i-1]
    
    f = np.divide(np.power(f,2),np.arange(1,m+1,1))
    
    imax = 0                    # 1 will be added later on
    fmax = f[0]                 # python is zero-based
    for i in range(1,m):
        if f[i] >= fmax:
            imax = i        
            fmax = f[i]
    
    for i in range(imax+1,m):
        x[indexsort[i]] = 0
        
    imax += 1                   # + 1 to correct imax

    return x, imax, fmax 
