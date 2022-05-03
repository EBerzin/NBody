# NBody

To run: python nbody.py -l [# of multigrid levels] -dt [time step]

To produce nbody.gif of particle positions: python nbody.py -l [# of multigrid levels] -dt [time step] -gif
Running with the -gif option increases execution time.

For description of arguments: python nbody.py -h

The code is currently configured to run a simulation with N = 256^2 particles, using EdS initial conditions generated using MUSIC.

