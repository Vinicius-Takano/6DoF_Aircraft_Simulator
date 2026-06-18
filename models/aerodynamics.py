#Aerodynamics.py
import numpy as np
import dynamics.kinematics as kin

def aero_force_coef_Fb(state, control, params):
    # unpack state
    _, q, _ = state[3:6]
    
    # unpack control
    aileron, elevator, rudder, _, _, _, _, _= control
    
    c = params["geometry"]["c"]
    
    # compute aerodynamic coefficients (placeholder)
    alpha, beta = kin.alpha_beta(state, control)
    V_a = kin.total_velocity(state, control)
    
    CL_0, CL_alpha, CL_q, CL_delta_e = params["aerodynamics"]["CL_0"], params["aerodynamics"]["CL_alpha"], params["aerodynamics"]["CL_q"], params["aerodynamics"]["CL_delta_e"]
    CD_0, k = params["aerodynamics"]["CD0"], params["aerodynamics"]["k"]
    Cy_beta, Cy_delta_a, Cy_delta_r = params["aerodynamics"]["Cy_beta"], params["aerodynamics"]["Cy_delta_a"], params["aerodynamics"]["Cy_delta_r"]

    CL = CL_0 + CL_alpha*alpha + CL_delta_e*elevator + CL_q*q*c/(2*V_a)
    CD = CD_0 + k*CL**2
    CYa = Cy_beta*beta + Cy_delta_a*aileron + Cy_delta_r*rudder
    
    C_Fw = np.array([-CD, CYa, -CL])
    C_Fb = np.linalg.inv(kin.rotation_matrix(alpha, beta, "body_to_wind")) @ C_Fw
    
    return C_Fb

def aero_moment_coef_Mb(state, control, params):
    # unpack state
    p, q, r = state[3:6]
    
    # unpack control
    aileron, elevator, rudder, _, _, _, _, _= control
    
    # unpack params
    c = params["geometry"]["c"]
    b = params["geometry"]["b"]
    
    # compute aerodynamic coefficients
    V_a = kin.total_velocity(state, control)
    alpha, beta = kin.alpha_beta(state, control)
    
    Cm_0, Cm_alpha, Cm_q, Cm_delta_e = params["aerodynamics"]["Cm_0"], params["aerodynamics"]["Cm_alpha"], params["aerodynamics"]["Cm_q"], params["aerodynamics"]["Cm_delta_e"]
    Cn_beta, Cn_delta_r, Cn_delta_a, Cn_e, Cn_r = params["aerodynamics"]["Cn_beta"], params["aerodynamics"]["Cn_delta_r"], params["aerodynamics"]["Cn_delta_a"], params["aerodynamics"]["Cn_e"], params["aerodynamics"]["Cn_r"]
    Cr_beta, Cr_delta_r, Cr_delta_a, Cr_e, Cr_r = params["aerodynamics"]["Cr_beta"], params["aerodynamics"]["Cr_delta_r"], params["aerodynamics"]["Cr_delta_a"], params["aerodynamics"]["Cr_e"], params["aerodynamics"]["Cr_r"]
    
    Cm = Cm_0 + Cm_alpha*alpha + Cm_delta_e*elevator + Cm_q*q*c/(2*V_a)
    Cn = Cn_beta*beta + Cn_delta_r*rudder + Cn_delta_a*aileron + Cn_r*r*b/(2*V_a) + Cn_e*p*b/(2*V_a)
    Cr = Cr_beta*beta + Cr_delta_r*rudder + Cr_delta_a*aileron + Cr_r*r*b/(2*V_a) + Cr_e*p*b/(2*V_a)
    
    return np.array([Cr, Cm, Cn])

def aero_force_Fb(state, control, params):
    C_Fb = aero_force_coef_Fb(state, control, params)
    Q = kin.dynamic_pressure(state, control)
    S_w = params["geometry"]["S_w"]
    
    F_b = C_Fb*Q*S_w
    
    return F_b

def aero_moment_Mb(state, control, params):
    C_Mb = aero_moment_coef_Mb(state, control, params)
    Q = kin.dynamic_pressure(state, control)
    S_w = params["geometry"]["S_w"]
    c = params["geometry"]["c"]
    b = params["geometry"]["b"]
    
    M_b = C_Mb*Q*S_w*np.array([b, c, b])
    
    return M_b

def aero_moment_cg(state, control, params):
    F_b = aero_force_Fb(state, control, params)
    M_b = aero_moment_Mb(state, control, params)
    
    c = params["geometry"]["c"]
    r_cg = np.array(params["geometry"]["r_cg"])
    r_ac = np.array(params["geometry"]["r_ac"])
    
    delta_ac = (r_cg - r_ac)*c
    
    M_cg = M_b + np.cross(delta_ac, F_b)
    
    return M_cg