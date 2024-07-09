"""
Created on Sat May 11 17:18:53 2024

@author: b5l
"""
import pandas as pd
import numpy as np
import pandas as pd
import re
import opendssdirect as dss
import os
from enum import Enum
import time
import datetime


class ControlType(Enum):
    WATT = 1
    VAR = 2
    WATT_VAR = 3

def get_matching_values(data, search_ids):
    """
    Return the list of values from 'data' where 'data.ids' match 'search_ids'.

    Parameters:
        data (X): An instance of class X containing 'ids' and 'values'.
        search_ids (list of str): IDs for which values are needed.

    Returns:
        list: A list of values corresponding to 'search_ids'.
    """
    id_value_map = dict(zip(data.ids, data.values))
    
    matched_values = [id_value_map[id] for id in search_ids if id in id_value_map]
    
    return matched_values

def loadflow_dss(opt_result, load_P, load_Q):
    """
    This function is used to calculate the result by call OpenDss

    """         
    ids = ['1.1', '2.2', '4.3', '5.3', '6.3', '7.1', '9.1', '10.1', '11.1', '12.2', '16.3', '17.3', 
               '19.1', '20.1', '22.2', '24.3', '28.1', '29.1', '30.3', '31.3', '32.3', '33.1', '34.3', 
               '35.1', '35.2', '37.1', '38.2', '39.2', '41.3', '42.1', '43.2', '45.1', '46.1', '47.1', 
               '47.2', '47.3', '48.1', '48.2', '48.3', '49.1', '49.2', '49.3', '50.3', '51.1', '52.1', 
               '53.1', '55.1', '56.2', '58.2', '59.2', '60.1', '62.3', '63.1', '64.2', '65.1', '65.2', 
               '65.3', '66.3', '68.1', '69.1', '70.1', '71.1', '73.3', '74.3', '75.3', '76.1', '76.2', 
               '76.3', '77.2', '79.1', '80.2', '82.1', '83.3', '84.3', '85.3', '86.2', '87.2', '88.1', 
               '90.2', '92.3', '94.1', '95.2', '96.2', '98.1', '99.2', '100.3', '102.3', '103.3', '104.3',
               '106.2', '107.2', '109.1', '111.1', '112.1', '113.1', '114.1']
      
    num_buses = len(ids)
    # %%Call OpenDSS to get the load flow result
    Dist_fileName = './master.dss'  # Distribution System File directory
    OpenDSSfileName = Dist_fileName
    dss.Text.Command("clear")
    dss.Text.Command('Compile (' + OpenDSSfileName + ')')

    # Setting load active/reactive power value
    Allloads = dss.Loads.AllNames()
        
    for load, kw, kvar in zip(Allloads, load_P, load_Q):
        dss.Loads.Name(load)
        dss.Loads.kW(kw)
        dss.Loads.kvar(kvar)
        
    PV_bus = dss.PVsystems.AllNames()
    PV_Qset = opt_result[0:len(PV_bus)]

    for pvsystem,  kvar in zip(PV_bus,PV_Qset):
        dss.PVsystems.Name(pvsystem)
        # dss.PVsystems.kW(PV_max)
        dss.PVsystems.kvar(kvar)        

    # solve load flow
    dss.Text.Command('Set mode = snapshot')
    dss.Text.Command('Solve')
    # Chekc converge or not
    solve_converged = dss.Solution.Converged()

    if solve_converged:
        # get voltage magnitude in pu (each node)
        Allnodename = dss.Circuit.AllNodeNames()
        AllVolMag = dss.Circuit.AllBusMagPu()  # Get all bus voltage magnitude in pu
        AllVolMag = np.array(AllVolMag)
        Total_losses = np.array(dss.Circuit.Losses())*1e-3
        real_loss = Total_losses[0]  # active power losses

        load_indexes = [Allnodename.index(item) for item in [n.lower() for n in list(ids)]]
        loadVolMag = [AllVolMag[index] for index in load_indexes]
        loadVolMag = np.array(loadVolMag)

        # Calculate penalties
        num_overvoltage = np.sum(loadVolMag > 1.05)
        num_undervoltage = np.sum(loadVolMag < 0.95)
        if num_overvoltage > 0 or num_undervoltage > 0:
        # if np.any((loadVolMag < 0.95) | (loadVolMag > 1.05)):
            output_obj = (num_overvoltage+num_undervoltage)*1000
            bus_vol = loadVolMag
        else:
            output_obj = real_loss
            bus_vol = loadVolMag

    else:
        output_obj = 10000
        bus_vol = np.ones((num_buses, 1))
        
    return output_obj, bus_vol    

def dopf_step(powers_real,powers_imag,t):
    
    # network = "IEEE123_input.csv"
    # para = pd.read_csv(network)  # Loading simulation parameters
    PV_profile = pd.read_csv("PV_vary_15min.txt", header=None)
    PV_profile = PV_profile.values

    #####################################################################

    ids = ['1.1', '2.2', '4.3', '5.3', '6.3', '7.1', '9.1', '10.1', '11.1', '12.2', '16.3', '17.3', 
               '19.1', '20.1', '22.2', '24.3', '28.1', '29.1', '30.3', '31.3', '32.3', '33.1', '34.3', 
               '35.1', '35.2', '37.1', '38.2', '39.2', '41.3', '42.1', '43.2', '45.1', '46.1', '47.1', 
               '47.2', '47.3', '48.1', '48.2', '48.3', '49.1', '49.2', '49.3', '50.3', '51.1', '52.1', 
               '53.1', '55.1', '56.2', '58.2', '59.2', '60.1', '62.3', '63.1', '64.2', '65.1', '65.2', 
               '65.3', '66.3', '68.1', '69.1', '70.1', '71.1', '73.3', '74.3', '75.3', '76.1', '76.2', 
               '76.3', '77.2', '79.1', '80.2', '82.1', '83.3', '84.3', '85.3', '86.2', '87.2', '88.1', 
               '90.2', '92.3', '94.1', '95.2', '96.2', '98.1', '99.2', '100.3', '102.3', '103.3', '104.3',
               '106.2', '107.2', '109.1', '111.1', '112.1', '113.1', '114.1']
    
   #####################################################################
    load_P = get_matching_values(powers_real, ids)
    load_Q = get_matching_values(powers_imag, ids)

    ######################################################################
    # PV parameters
    PV_bus = ['7', '29', '49', '55', '63', '68', '80', '87', '113']
    PV_number = len(PV_bus)
    # PV_capacity = para['PV_capacity'].dropna().values    
    PV_capacity = 150
  
    PV_P = PV_capacity*PV_profile
    PV_Qmax = np.sqrt(PV_capacity**2-PV_P**2)
    PV_Qmin = -PV_Qmax
    
    # Optimization Settings
    max_iterations = 30
    swarm_size = 30
    num_variables = PV_number

    #  PSO iteration
     
    inertia_weight = 0.9  #1
    cognitive_weight = 1.5 #1
    social_weight = 1.5 #3
    num_particles = int(swarm_size)
    num_iterations = int(max_iterations)


    bounds = list()
    for i in range(PV_number):
        bounds.append((PV_Qmin[t], PV_Qmax[t]))        

    # Initialize particles position and velocity
    particles_position = np.zeros((num_particles, num_variables))

    for i in range(num_variables):
        particles_position[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], num_particles)
    particles_velocity = np.random.uniform(-1, 1, size=(num_particles, num_variables))

      # Initialize particle best positions and global best position
    particles_best_position = particles_position.copy()
    # would revise particles best fitness
    particles_best_cost = np.zeros(num_particles)
    for i in range(num_particles):
        particles_best_cost[i] = loadflow_dss(particles_best_position[i], load_P, load_Q)[0]  # would replace this line

    global_best_cost = particles_best_cost.min()
    global_best_position = particles_best_position[particles_best_cost.argmin()]


    # PSO loop
    for iter in range(num_iterations):
        # print("Iteration number:{} Time step:{}".format(iter + 1, t))
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(num_variables), np.random.rand(num_variables)
            cognitive_velocity = cognitive_weight * r1 * \
                (particles_best_position[i] - particles_position[i])
            social_velocity = social_weight * r2 * \
                (global_best_position - particles_position[i])
            particles_velocity[i] = inertia_weight * \
                particles_velocity[i] + cognitive_velocity + social_velocity

            # Update position
            particles_position[i] += particles_velocity[i]

            # Clip position to be within bounds
            for d in range(num_variables):
                # particles_position[i, d] = np.clip(particles_position[i, d], bounds[d][0], bounds[d][1])
                a = np.clip(particles_position[i, d], bounds[d][0], bounds[d][1])
                particles_position[i, d] = a[0]

            # Update particle best position
            particle_new_cost = loadflow_dss(particles_position[i], load_P, load_Q)[0]
            if particle_new_cost < particles_best_cost[i]:
                particles_best_position[i] = particles_position[i]
                particles_best_cost[i] = particle_new_cost

            # Update global best position
            if particles_best_cost[i] < global_best_cost:
                global_best_cost = particles_best_cost[i]
                global_best_position = particles_best_position[i]
          
    converted_pv_bus = ['PVSystem.' + str(bus).split('.')[0] for bus in PV_bus]        
    # output dictionary
    # "values": (global_best_position[0:PV_number]).tolist(), 
    #  "values": (np.zeros(PV_number)).tolist(), 
    PV_reactive = {
        "values": (global_best_position[0:PV_number]).tolist(), 
        "eqid": converted_pv_bus,
        "units": "VAR"
    }
    cost, voltage_v = loadflow_dss(global_best_position, load_P, load_Q)
    print(cost)
    Bus_voltages = {"values": voltage_v.tolist(), 
        # "ids": [bus.upper() for bus in load_buses],
        "ids": ids,
        "units": "kV"}
    return PV_reactive, Bus_voltages


