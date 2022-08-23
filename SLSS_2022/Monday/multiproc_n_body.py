#!/usr/bin/env python
import multiprocessing
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def getAcc(pos, mass, mass_sq, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    N = pos.shape[0]
    a = np.zeros((N, 3))
    for i in range(N):
        dr_all = pos - pos[i]
        inv_r3_all = (np.sum(dr_all ** 2, axis=1) + softening ** 2) ** (-1.5)
        a[i] = G * np.dot(dr_all.T, inv_r3_all * mass_sq)
    return a


def getEnergy(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE


def run_sim_mc(procnum, return_dict):
    """ N-body simulation """

    print("Running simulation")
    # Simulation parameters
    N = 100  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 10.0  # time at which simulation ends
    dt = 0.01  # timestep
    softening = 0.1  # softening length
    G = 1.0  # Newton's Gravitational Constant

    # Generate Initial Conditions
    np.random.seed(procnum)  # set the random number generator seed

    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    mass_sq = np.squeeze(mass)
    pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    vel = np.random.randn(N, 3)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, mass_sq, G, softening)

    # calculate initial energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos
    KE_save = np.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = np.arange(Nt + 1) * dt

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, mass_sq, G, softening)
    
    return_dict[procnum] = acc

    
# def run_sim_mc(procnum, return_dict):
#     print("Running sleep simulation")
#     time.sleep(5)

    
# def run_sim_mc(procnum, return_dict):
#     print("Running sleep simulation")
#     time.sleep(np.random.randint(1, 5))


if __name__ == '__main__':
    ###### Multiple core processing #########
    time_start = time.time()
    
    # Get number of processor cores
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    # Get number of processor cores
    num_of_sims = os.cpu_count()
#     num_of_sims = 16
    print(f"Running simulations on {num_of_sims} cores with multiprocessing.")

    # Create pool of workers
    pool = multiprocessing.Pool(num_of_sims)

    # Map pool of workers to process
    pool.starmap(func=run_sim_mc, iterable=[(i, return_dict) for i in range(num_of_sims)])

    # Wait until workers complete execution
    pool.close()
    time_end = time.time()
    print(f"Time elapsed: {round(time_end - time_start, 2)}s")
    
    
    ###### Single core processing #########
    print(f"Running simulations on single core.")
    time_start = time.time()
    
    return_dict = {}

    for i in range(num_of_sims):
        run_sim_mc(i, return_dict)

    time_end = time.time()
    print(f"Time elapsed: {round(time_end - time_start, 2)}s")



