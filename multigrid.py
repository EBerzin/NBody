# Multigrid

import numpy as np
import time

class Grid:
    omega = 2/3
    def __init__(self,n):
        self.n = n
        self.h =  1/n
        self.uold = np.zeros([n+2,n+2])
        self.unew = np.zeros([n+2,n+2])
        self.res = np.zeros([n+2,n+2])
        self.err = np.zeros([n,n])
        self.rhs = np.zeros([n,n])


def boundary(uu,bc):

    if(bc=='dirichlet'):
        uu[0,1:-1] = -uu[1,1:-1]
        uu[-1,1:-1] = -uu[-2,1:-1]
        uu[1:-1,0] = -uu[1:-1,1]
        uu[1:-1,-1] = -uu[1:-1,-2]
        uu[0,0] = -uu[1,1]
        uu[-1,0] = -uu[-2,1]
        uu[0,-1] = -uu[1,-2]
        uu[-1,-1] = -uu[-2,-2]

    if(bc=='neumann'):
        uu[0,1:-1] = uu[1,1:-1]
        uu[-1,1:-1] = uu[-2,1:-1]
        uu[1:-1,0] = uu[1:-1,1]
        uu[1:-1,-1] = uu[1:-1,-2]
        uu[0,0] = uu[1,1]
        uu[-1,0] = uu[-2,1]
        uu[0,-1] = uu[1,-2]
        uu[-1,-1] = uu[-2,-2]

    if(bc=='periodic'):
        uu[0,1:-1] = uu[-2,1:-1]
        uu[-1,1:-1] = uu[1,1:-1]
        uu[1:-1,0] = uu[1:-1,-2]
        uu[1:-1,-1] = uu[1:-1,1]
        uu[0,0] = uu[-2,-2]
        uu[-1,0] = uu[1,-2]
        uu[0,-1] = uu[-2,1]
        uu[-1,-1] = uu[1,1]

def jacobi(mg,l,niter,bc):

    g = mg[l-1]

    for iter in range(0,niter):

        g.unew[1:-1,1:-1] = g.uold[1:-1,1:-1] + 0.25 * g.omega * \
        (g.uold[2:,1:-1]+g.uold[:-2,1:-1]+g.uold[1:-1,2:]+g.uold[1:-1,:-2]-4.0*g.uold[1:-1,1:-1] \
         + g.h**2 * g.rhs )

        g.err[:,:] = g.uold[1:-1,1:-1] - g.unew[1:-1,1:-1]

        g.uold[1:-1,1:-1] = g.unew[1:-1,1:-1]

        boundary(g.uold,bc)

def residual(mg,l,bc):

    g = mg[l-1]

    g.res[1:-1,1:-1]= \
        ((g.uold[2:,1:-1]+g.uold[:-2,1:-1]+g.uold[1:-1,2:]+g.uold[1:-1,:-2]-4.0*g.uold[1:-1,1:-1])/g.h**2 \
         + g.rhs )

    boundary(g.res,bc)

def restriction(mg,l):

    fine=mg[l-1]

    coarse=mg[l-2]

    coarse.rhs[:,:]=0

    coarse.rhs[:,:]=coarse.rhs[:,:]+1./64*fine.res[0:-3:2,0:-3:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[1:-2:2,0:-3:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[2:-1:2,0:-3:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+1./64*fine.res[3:  :2,0:-3:2]

    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[0:-3:2,1:-2:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+9./64*fine.res[1:-2:2,1:-2:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+9./64*fine.res[2:-1:2,1:-2:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[3:  :2,1:-2:2]

    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[0:-3:2,2:-1:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+9./64*fine.res[1:-2:2,2:-1:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+9./64*fine.res[2:-1:2,2:-1:2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[3:  :2,2:-1:2]

    coarse.rhs[:,:]=coarse.rhs[:,:]+1./64*fine.res[0:-3:2,3:  :2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[1:-2:2,3:  :2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+3./64*fine.res[2:-1:2,3:  :2]
    coarse.rhs[:,:]=coarse.rhs[:,:]+1./64*fine.res[3:  :2,3:  :2]

    coarse.uold[:,:]=0

def prolongation(mg,l,bc):

    fine=mg[l-1]

    coarse=mg[l-2]

    fine.err[:,:]=0

    fine.err[0:-1:2,0:-1:2]=fine.err[0:-1:2,0:-1:2]+1/16*coarse.uold[0:-2,0:-2]
    fine.err[0:-1:2,0:-1:2]=fine.err[0:-1:2,0:-1:2]+3/16*coarse.uold[1:-1,0:-2]
    fine.err[0:-1:2,0:-1:2]=fine.err[0:-1:2,0:-1:2]+3/16*coarse.uold[0:-2,1:-1]
    fine.err[0:-1:2,0:-1:2]=fine.err[0:-1:2,0:-1:2]+9/16*coarse.uold[1:-1,1:-1]

    fine.err[1:  :2,0:-1:2]=fine.err[1:  :2,0:-1:2]+3/16*coarse.uold[1:-1,0:-2]
    fine.err[1:  :2,0:-1:2]=fine.err[1:  :2,0:-1:2]+1/16*coarse.uold[2:  ,0:-2]
    fine.err[1:  :2,0:-1:2]=fine.err[1:  :2,0:-1:2]+9/16*coarse.uold[1:-1,1:-1]
    fine.err[1:  :2,0:-1:2]=fine.err[1:  :2,0:-1:2]+3/16*coarse.uold[2:  ,1:-1]

    fine.err[0:-1:2,1:  :2]=fine.err[0:-1:2,1:  :2]+3/16*coarse.uold[0:-2,1:-1]
    fine.err[0:-1:2,1:  :2]=fine.err[0:-1:2,1:  :2]+9/16*coarse.uold[1:-1,1:-1]
    fine.err[0:-1:2,1:  :2]=fine.err[0:-1:2,1:  :2]+1/16*coarse.uold[0:-2,2:  ]
    fine.err[0:-1:2,1:  :2]=fine.err[0:-1:2,1:  :2]+3/16*coarse.uold[1:-1,2:  ]

    fine.err[1:  :2,1:  :2]=fine.err[1:  :2,1:  :2]+9/16*coarse.uold[1:-1,1:-1]
    fine.err[1:  :2,1:  :2]=fine.err[1:  :2,1:  :2]+3/16*coarse.uold[2:  ,1:-1]
    fine.err[1:  :2,1:  :2]=fine.err[1:  :2,1:  :2]+3/16*coarse.uold[1:-1,2:  ]
    fine.err[1:  :2,1:  :2]=fine.err[1:  :2,1:  :2]+1/16*coarse.uold[2:  ,2:  ]

    fine.uold[1:-1,1:-1]=fine.uold[1:-1,1:-1]+fine.err[:,:]

    boundary(fine.uold,bc)

def vcycle(mg,l,bc):
    if(l == 1):
        jacobi(mg,l,10,bc)
    else:
        jacobi(mg,l,3,bc)
        residual(mg,l,bc)
        restriction(mg,l)
        vcycle(mg,l-1,bc)
        prolongation(mg,l,bc)
        jacobi(mg,l,3,bc)

def multigrid(mg,l,itermax, convmg, bc,eps):

    iter = 0

    norm = 1.0

    start = time.time()

    n = 2**l

    while (iter < itermax and norm > eps):

        vcycle(mg,l,bc)

        norm = np.max(abs(mg[l-1].err))*n*n

        iter = iter + 1

        convmg[iter] = norm

    end = time.time()
