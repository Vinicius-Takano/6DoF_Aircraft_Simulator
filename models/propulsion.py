# Propulsion.py
import numpy as np
import dynamics.equations as eq
import models.atmosphere as atm

def propulsive_effects(state, control, params):
    throttle_1, throttle_2 = control[3:5]
    T_max = params["propulsion"]["T_max"]
    engine_i = np.radians(params["geometry"]["engine_i"])
    rho_sea_level = params["atmosphere"]["rho"]
    
    # compute thrust from throttle settings
    rho, _ = atm.compute_atmospheric_properties(state) 
    T1 = throttle_1*T_max*(rho/rho_sea_level)       #will add rho dependence later
    T2 = throttle_2*T_max*(rho/rho_sea_level)       #will add rho dependence later
    
    F_T = np.array([(T1 + T2)*np.cos(engine_i), 0, -(T1 + T2)*np.sin(engine_i)])  #total thrust in body frame (inclined by engine incidence)
    
    # moments from thrust
    _, _, _, _, engine_positions = eq.aicraft_geometry(params["geometry"])
    
    T_vec_1 = np.array([T1*np.cos(engine_i), 0, -T1*np.sin(engine_i)])
    T_vec_2 = np.array([T2*np.cos(engine_i), 0, -T2*np.sin(engine_i)])
    
    M_1 = np.cross(engine_positions[:,0], T_vec_1)   
    M_2 = np.cross(engine_positions[:,1], T_vec_2)  
    
    M_engine = M_1 + M_2                    # engine moment around center of mass
    
    return F_T, M_engine
