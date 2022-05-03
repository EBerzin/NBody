import numpy as np
import matplotlib.pyplot as plt
import time
from multigrid import *
from PIL import Image, ImageDraw
import io
import matplotlib as mpl
import argparse

mpl.rcParams['figure.dpi']= 600


def getArgs() :
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--levels", type=int, help = "Number of multigrid levels, producing a fine grid of 2^l.")
    parser.add_argument("-dt", "--dt", type=float, help = "Time step size.")
    parser.add_argument("-gif", "--gif",action="store_true", help = "Output GIF of simulation (nbody.gif).")
    return parser.parse_args()


class Cosmology:

    def __init__(self, H0, O_M, O_L):
        self.H0 = H0
        self.O_M = O_M
        self.O_L = O_L

    def O_K(self):
        return 1 - self.O_M - self.O_L

    def da(self, a):
        return self.H0 * a * np.sqrt(self.O_L + self.O_M * a**-3 + self.O_K() * a**-2)

# Time integration Functions
def dXda(p, a, cosmology):
    da = cosmology.da(a)
    return p * cosmology.H0 / (a**2 * da)

def dPda(del_phi_x, del_phi_y, positions, N_grid, L, a, cosmology):
    particle_x = interp(N_grid, L, positions, del_phi_x)
    particle_y = interp(N_grid, L, positions, del_phi_y)

    del_phi = np.c_[particle_x, particle_y]
    da = cosmology.da(a)
    return -del_phi * cosmology.H0/da

def leapfrog(X, P, del_phi_x, del_phi_y, a, cosmology, N_grid, L,dt):
    P += dPda(del_phi_x, del_phi_y, X, N_grid, L, a, cosmology) * dt
    a = a + dt/2
    X += dXda(P, a, cosmology) * dt
    return X, P

def gradient(phi,L):
    N = len(phi)
    h = L/N
    phi_val = np.zeros((N+4,N+4))
    phi_val[2:N+2,2:N+2] = phi

    # Enforce Periodicity
    phi_val[:,N+2] = phi_val[:,2]
    phi_val[:,N+3] = phi_val[:,3]
    phi_val[:,1] = phi_val[:,N+1]
    phi_val[:,0] = phi_val[:,N]

    phi_val[N+2,:] = phi_val[2,:]
    phi_val[N+3,:] = phi_val[3,:]
    phi_val[1,:] = phi_val[N+1,:]
    phi_val[0,:] = phi_val[N,:]


    partial_x = (-phi_val[2+2:,2:N+2] + 8 * phi_val[2+1:N+3,2:N+2] - 8 * phi_val[2-1:N+1,2:N+2] + phi_val[2-2:N,2:N+2]) / (12 * h)
    partial_y = (-phi_val[2:N+2,2+2:] + 8 * phi_val[2:N+2,2+1:N+3] - 8 * phi_val[2:N+2,2-1:N+1] + phi_val[2:N+2,2-2:N]) / (12 * h)
    return partial_x, partial_y

# Particle mesh Functions

# Creating the density field using the CIC method
def CIC(Ng, L, X, vals =[]):

    # Ng : number of cells in grid
    # L : extent of grid
    # X : particle positions

    rho = np.zeros((Ng,Ng))

    h = L/Ng

    pf = X[:,0]/h
    qf = X[:,1]/h

    p = np.floor(pf).astype('int')
    q = np.floor(qf).astype('int')

    pc = pf - p
    qc = qf - q

    for i in range(len(X)):
        rho[p[i]%Ng,q[i]%Ng] += (1-pc[i])*(1-qc[i]) * vals[i]
        rho[(p[i]+1)%Ng, q[i]%Ng] += pc[i] * (1-qc[i]) * vals[i]
        rho[p[i]%Ng,(q[i]+1)%Ng] += (1-pc[i]) * qc[i] * vals[i]
        rho[(p[i]+1)%Ng,(q[i]+1)%Ng] += pc[i] * qc[i] * vals[i]

    return rho

# Interpolating back to particles
def interp(Ng, L, X, vals):

    # Ng : number of cells in grid
    # L : extent of grid
    # X : particle positions
    # vals : values to interpolate

    particle_vals = np.zeros(len(X))

    h = L/Ng

    pf = X[:,0]/h
    qf = X[:,1]/h

    p = np.floor(pf).astype('int')
    q = np.floor(qf).astype('int')

    pc = (pf - p)
    qc = (qf - q)


    for i in range(len(X)):
        particle_vals[i] += vals[p[i]% Ng,q[i]% Ng] * ((1-pc[i])*(1-qc[i]))
        particle_vals[i] += vals[(p[i]+1)%Ng, q[i]% Ng] * (pc[i] * (1-qc[i]))
        particle_vals[i] += vals[p[i]% Ng,(q[i]+1)%Ng] * ((1-pc[i]) * qc[i])
        particle_vals[i] += vals[(p[i]+1)%Ng,(q[i]+1)%Ng] * (pc[i] * qc[i])

    return particle_vals


if __name__ == "__main__":

    args = getArgs()

    LCDM = Cosmology(68.0, 0.31, 0.69)
    EdS = Cosmology(70.0, 1.0, 0.0)

    N = 256
    N_particles = N**2
    L = 50
    h = L
    vc = h * EdS.H0

    positions = np.zeros((N_particles,2))
    momenta = np.zeros((N_particles,2))
    positions = np.indices((N,N), dtype = 'float64').transpose(1,2,0).reshape(N_particles,2) * (L/N)

    dx = np.load('initial_x.npy').reshape(N_particles)
    dy = np.load('initial_y.npy').reshape(N_particles)
    vx = np.load('initial_px.npy').reshape(N_particles)
    vy = np.load('initial_py.npy').reshape(N_particles)

    for i in range(N_particles-1):
        positions[i,:] += [dx[i] , dy[i]]
        momenta[i,:] = [vx[i] / vc, vy[i] /  vc]

    positions = positions / h


    # Initializing multigrid parameters
    lmax = args.levels
    N_grid = 2**lmax
    bc='periodic'

    a = 0.02
    dt = args.dt
    mass = (N_grid / np.sqrt(N_particles))**2
    counter = 0

    images = []

    while (a < 2):
        print('a = %.2f'%(a))

        if (args.gif):

            fig = plt.figure()
            plt.scatter(positions[:,0],positions[:,1], s = 1)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('$\\tilde{x}$')
            plt.ylabel('$\\tilde{y}$')
            plt.title('a = %.2f'%(a))
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            im = Image.open(img_buf)
            images.append(im)
            plt.close(fig)

        counter += 1

        # Calculating density
        rho = CIC(N_grid,L/h, positions, np.ones(len(positions)) * mass)

        # Performing multigrid
        delta = rho - 1
        f = delta
        eps = 1e-10
        itermax = 1000

        mg=[]
        for l in range(1,lmax+1):
            n=2**l
            mg.append(Grid(n))

        mg[lmax-1].rhs[:,:] = -f[:,:]
        mg[lmax-1].uold[:,:] = 0

        convmg = np.zeros(itermax+1)
        convmg[0] = np.max(abs(mg[lmax-1].rhs[:,:]))

        multigrid(mg,lmax,itermax, convmg, bc,eps)

        phi = mg[lmax-1].uold[1:-1,1:-1] * ((3/2) * EdS.O_M  /a)

        # Finding the gradient of phi
        del_phi_x, del_phi_y = gradient(phi,L/h)

        # Updating particle positions and momenta
        positions, momenta = leapfrog(positions, momenta, del_phi_x, del_phi_y, a, EdS, N_grid, L/h, dt)

        a += dt

    if (args.gif):
        print('Saving GIF')
        images[0].save('nbody.gif', save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
