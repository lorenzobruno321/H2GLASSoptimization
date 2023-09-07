# coding
"""

Created on Tue 11/07/2023 12:48
#====================================================================
## #[Design]
#====================================================================
# @Coded by     : Lorenzo Bruno
# @email        : lorenzo.bruno@studenti.polito.it
# @Date         : July, 2023
# @Dynamic Model
# @Paper        : UPC collaboration

"""

#import numpy as np
import globalPARAMETERS_LB as par
#import time
import pyomo.environ as pyomo
from pyomo.core import maximize
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
#from globalPARAMETERS_LB import*
import matplotlib.pyplot as plt




##
#=========================================================================|
# function[output arguments] = new_model_OPTIMIZATION(input paramaters)   |
#=========================================================================|
# #[Design]                                                               |
#=========================================================================|
# @coded by   : Lorenzo Bruno                                             |
# @email      : lorenzo.bruno@studenti.polito.it                          |
# @Time Scale : hours                                                     |
# @Dyn Model  :                                                           |
# @Paper      : UPC collaboration                                         |
# @author     :                                                           |
# @title      :                                                           |
#=========================================================================|
## #[Description]                                                         |
#=========================================================================|
# modelling and optimal control of hydrogen-based storage system in one year connected to glass furnace
# Problem 1.1) is solved by minimizing the overall cost function
# The goal is to minimize the cost function related to the component costs and load tracking.
#=========================================================================|
## [Function]                                                             |
#=========================================================================|
#        (1.1) new_model_OPTIMIZATION 				                      |
#=========================================================================|
# # set up the optimization problem (1.1)                                 |
#=========================================================================|
#-------------------------(problem (1.1))---------------------------------|
#=========================================================================|
#        min (sum costs)		         			                      |
#        s.t. 		      						                          |
#        H(k+1) =  H[k] + Pin[k]*deltat - Pout[k]*deltat         |
#        Logical states constraints	                		              |
#        Mode transitions constraints                              	      |
#        Operating constraints					                          |
#        Ramp Up   constraints                     			              |
#        Physical  constraints,				             	              |
#        Power balance constraints,                                       |
#=========================================================================|
# #Parameters  : INPUT parameters                                         |
#=========================================================================|
#     list_pv           Power forecast of PV generation  	    	      |
#     load_furnace         Power load at furnace 	    	              |
#     loh_ht        Initial level of hydrogen storage (H(0)).  .          |
#=========================================================================|
# #Return  : OUTPUT arguments			                                  |
#=========================================================================|
#       power_in_ele    power inlet in the ELECTROLYSER                   |
#       power_out_ele   power outlet in the ELECTROLYSER                  |
#       power_out_ele_bur power outlet of the ELECTROLYSER to BURNER             |
#       power_out_ele_cp power outlet in the ELECTROLYSER to COMPRESSOR                 |
#       optDELTA_ele    DELTA function                                     |
#       power_OPT_ele   power of optimization                              |
#       optZON_ele      auxiliary variable                                 |
#       power_pv        power PV for the supply             |
#       power_grid      power GRID for the supply     |
#       power_in_cp     power inlet at COMPRESSOR                    |
#       power_in_ht     power inlet at STORAGE TANK                    |
#       power_out_ht    power outlet at STORAGE TANK                    |
#       capacity_ht     CAPACITY [kWh] of STORAGE TANK                    |
#       power_out_bo    power output at STORAGE BOTTLE                      |
#       capacity_bo     CAPACITY [kWh] of STORAGE BOTTLE                    |
#       power_in_bur    power inlet at BURNER                               |
#       power_out_bur   power outlet at BURNER                              |
#=========================================================================|
## Notes      									                          |
#=========================================================================|
# Problem 1.1) The main goal of optimal control is to track the requested load
#=========================================================================|
## Prerequites                                                            |
# =========================================================================|
#   [1]. CVXPY : https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions-for-non-standard-platforms
#   [2]. GUROBI: https://www.gurobi.com/documentation/9.0/quickstart_windows/ins_the_anaconda_python_di.html#section:Anaconda
    # ====================================================================================================================
    #-------------------- Free Variables -------------------------------------
    # ====================================================================================================================
    ## # Definition of SET VECTORS using PYOMO tools
    # ====================================================================

model = pyomo.ConcreteModel()                      #AbstractModel();


#model.t = pyomo.Set(initialize=list(range(0, par.time_end)))
#model.dt = pyomo.Set(initialize=list(range(0, par.time_end+1)))
model.t = pyomo.Set(initialize=range(0, par.time_end))
model.dt = pyomo.Set(initialize=range(0, par.time_end+1))

# ====================================================================
## ELECTROLYSER
# ====================================================================
model.power_in_ele = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.power_out_ele = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.power_out_ele_bur = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.power_out_ele_cp = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)

model.optDELTA_ele = pyomo.Var(model.t, domain=pyomo.Binary, initialize=0)
model.power_OPT_ele = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.optZON_ele = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
#model.optGAMMA = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
# ====================================================================
## PV SUPPLY
# ====================================================================
model.list_pv = pyomo.Param(model.t, initialize=par.PV)
model.power_pv = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
# ====================================================================
## FURNACE LOAD
# ====================================================================
model.list_load_furnace = pyomo.Param(model.t, initialize=par.PL)
# ====================================================================
## GRID SUPPLY
# ====================================================================
model.power_grid = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
# ====================================================================
## COMPRESSOR
# ====================================================================
model.power_in_cp = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
# ====================================================================
## HYDROGEN STORAGE TANK
# ====================================================================
model.power_in_ht = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.power_out_ht = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.capacity_ht = pyomo.Var(model.dt, domain=pyomo.Reals, initialize=0)
# ====================================================================
## HYDROGEN BOTTLE TANK
# ====================================================================
model.power_out_bo = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.capacity_bo = pyomo.Var(model.dt, domain=pyomo.Reals, initialize=0)
# ====================================================================
## BURNER
# ====================================================================
model.power_in_bur = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)
model.power_out_bur = pyomo.Var(model.t, domain=pyomo.Reals, initialize=0)

# ===============================================================================================================================================
## # Definition of CONSTRAINTS
# ================================================================================================================================================

model.constraints = pyomo.ConstraintList()

#k is cycle over the 1 year
for k in range(0, par.time_end):
    if k > 0:
        #ELECTROLYSER
        ZON_ele_prev = model.optZON_ele[k-1]
        
    else:
        # ELECTROLYSER
        ZON_ele_prev = 0
        model.capacity_ht[k] == par.loh_ht*par.capacity_ht_rated
        
        #BOTTLE TANK
        model.capacity_bo[k] = par.capacity_bo_rated
        
    # =========================================================================
    ## eq1: PV GENERATION DEFINITION
    # ==========================================================================
    model.constraints.add(model.power_pv[k] == par.cap_installed_pv*model.list_pv[k])            
    # =========================================================================
    ## eq2: POWER BALANCE node1
    # ==========================================================================
    model.constraints.add(model.power_grid[k] + model.power_pv[k] == model.power_in_ele[k]+ model.power_in_cp[k])
    # =========================================================================
    ## eq3: ELECTROLYSER EFFICIENCY
    # ==========================================================================
    model.constraints.add(model.power_out_ele[k] == par.efficiency_ele * model.power_in_ele[k])
    # =========================================================================
    ## eq4: LINEARIZATION of the PROBLEM: power_OPT_ele definition
    # ==========================================================================
    model.constraints.add(model.power_OPT_ele[k] == model.power_in_ele[k] - model.power_out_ele[k])
    # =========================================================================
    ## eq5: LINEARIZATION of the PROBLEM: Z definition
    # ==========================================================================
    model.constraints.add(model.optZON_ele[k] == model.optDELTA_ele[k] * model.power_OPT_ele[k])
    # =========================================================================
    ## eq6: LINEARIZATION of the PROBLEM: constraint on Z
    # ==========================================================================
    model.constraints.add(-par.M*model.optDELTA_ele[k] <= model.optZON_ele[k])
    model.constraints.add(model.optZON_ele[k] <= par.M*model.optDELTA_ele[k])
    # =========================================================================
    ## eq7: LINEARIZATION of the PROBLEM: constraint on Z - power_OPT_ele
    # ==========================================================================
    model.constraints.add(-par.M*(1-model.optDELTA_ele[k]) <= model.optZON_ele[k] - model.power_OPT_ele[k])
    model.constraints.add(model.optZON_ele[k] - model.power_OPT_ele[k] <= par.M*(1-model.optDELTA_ele[k]))
    # =========================================================================
    ## eq8: RAMP UP equation
    # ==========================================================================
    #model.constraints.add(abs(model.power_out_ele[k] - power_prev_ele) <= par.val_rampup)
    model.constraints.add(model.optZON_ele[k] - ZON_ele_prev <= par.val_rampup)
    model.constraints.add(-(model.optZON_ele[k] - ZON_ele_prev) <= par.val_rampup)
    # =========================================================================
    ## eq9: POWER BALANCE node2
    # ==========================================================================
    model.constraints.add(model.power_out_ele[k] == model.power_out_ele_cp[k] + model.power_out_ele_bur[k])
    # =========================================================================
    ## eq10: POWER required by the COMPRESSOR
    # ==========================================================================
    model.constraints.add(model.power_in_cp[k] == model.power_out_ele_cp[k]*par.compression_work/par.LHV_h2)
    # =========================================================================
    ## eq11: COMPRESSOR POWER BALANCE
    # ==========================================================================
    model.constraints.add(model.power_out_ele_cp[k] == model.power_in_ht[k])
    # =========================================================================
    ## eq12: ENERGY BALANCE in the STORAGE TANK
    # ==========================================================================
    model.constraints.add(model.capacity_ht[k+1] == model.capacity_ht[k] + model.power_in_ht[k] - model.power_out_ht[k])                #in terms of ENERGy [kWh], where POWER is multiplied by hour time step
    # =========================================================================
    ## eq13: STORAGE TANK initial condition --> condition at t =0
    # ==========================================================================
    #model.constraints.add(model.capacity_ht[0] == par.loh_ht*par.capacity_ht_rated)    #is in the initial IF CONDITION BLOCK
    # =========================================================================
    ## eq14: ENERGY BALANCE in the STORAGE TANK : CAPACITY CONSTRAINT
    # ==========================================================================
    model.constraints.add(par.perc_min_ht * par.capacity_ht_rated <= model.capacity_ht[k])          # we have to consider the E[k+1] or E[k]?
    model.constraints.add(model.capacity_ht[k] <= par.perc_max_ht * par.capacity_ht_rated)
    # =========================================================================
    ## eq15: ENERGY BALANCE in the STORAGE BOTTLE
    # ==========================================================================
    ############model.constraints.add(model.capacity_bo[k+1] == model.capacity_bo[k] - model.power_out_bo[k])
    # =========================================================================
    ## eq16: STORAGE BOTTLE initial condition --> condition at t =0 
    # ==========================================================================
    ############model.constraints.add(model.capacity_bo[0] == par.capacity_bo_rated)
    # =========================================================================
    ## eq17: POWER BALANCE node3
    # ==========================================================================
    model.constraints.add(model.power_out_ele_bur[k] + model.power_out_ht[k] == model.power_in_bur[k] )             #+ model.power_out_bo[k]
    # =========================================================================
    ## eq18: BURNER EFFICIENCY
    # ==========================================================================
    model.constraints.add(model.power_out_bur[k] == par.efficiency_bur*model.power_in_bur[k])               ############model.constraints.add(model.power_out_bur[k] == par.efficiency_bur*model.power_in_bur[k])               ############model.constraints.add(model.power_out_bur[k] == model.power_in_bur[k])              #par.efficiency_bur *  
    # =========================================================================
    ## eq19: BURNER BALANCE: constraint on POWER AT INLET
    # ==========================================================================
    ############model.constraints.add(par.perc_min_bur*par.power_bur_rated <= model.power_in_bur[k])
    ############model.constraints.add(model.power_in_bur[k] <= par.perc_max_bur*par.power_bur_rated)
    # =========================================================================
    ## eq20: POWER BALANCE: LOAD CONSTRAINT
    # ==========================================================================
    model.constraints.add(model.power_out_bur[k] == model.list_load_furnace[k])


# ===============================================================================================================================================
## # COST FUNCTION of ELECTROLYSER AND STORAGES
# ===============================================================================================================================================

def obj_func(model):
    return sum(model.power_out_ele[kk]*par.OPEX_ele for kk in model.t)

'''
def obj_func(model):
    tot_CAPEX = par.power_rated_ele*par.CAPEX_ele + par.flow_rate_rated_ele*par.CAPEX_ht + par.power_rated_cp*par.CAPEX_cp
    tot_OPEX = (sum(model.power_out_ele[kk]*par.OPEX_ele for kk in model.t) + sum(model.power_in_cp[kk]*par.OPEX_cp for kk in model.t) + sum(model.power_in_ht[kk]/par.LHV_h2/3600*par.OPEX_cp for kk in model.t) + sum(model.power_out_bo[kk]/par.LHV_h2/3600*par.OPEX_bo for kk in model.t) + sum(model.power_out_bur[kk]*par.OPEX_bur for kk in model.t))*par.life
    tot_OPEX_disc = sum(tot_OPEX/(1 + par.disc_rate)**nn for nn in range(0, par.life))
    tot_SUPPLY = sum(model.power_grid[kk]*par.cost_energy_grid for kk in model.t)*par.life + par.cap_installed_pv*par.CAPEX_pv + sum(model.power_pv[kk]*par.OPEX_pv for kk in model.t)*par.life
    tot_INSTALL = par.power_rated_ele*par.INSTALL_ele
    tot_REPLACE = 4*par.power_rated_ele*par.REPLACE_ele         # REPLACE after 4 years
    return tot_CAPEX + tot_OPEX_disc + tot_SUPPLY + tot_INSTALL + tot_REPLACE
'''
    
model.obj = pyomo.Objective(rule=obj_func, sense=pyomo.minimize)
instance = model.create_instance()
opt = pyomo.SolverFactory('gurobi')
result = opt.solve(instance)
result.write()



# ===============================================================================================================================================
## # STORE the values model.optimization
# ===============================================================================================================================================
pow_out_ele = []
for ii in model.t:
    pow_out_ele.append(pyomo.value(model.power_out_ele[ii]))
        
plt.plot(model.t, pow_out_ele)
print('Result from OPTIMIZATION FUNCTION = ', model.obj())   





'''
outPower_elz = []
for ii in range(par.time_end):
    outPower_elz += [power_out_ele[ii].value]
        #outPower_elz.append(pyo.value(model.power_out_ele[ii]))
        #OPTDELTA_ELE.append(pyo.value(model.optDELTA_ele[ii]))




        else:
            # ELECTROLYSER
            power_prev_ele = power_0_ele
            # STORAGE TANK            
            model.capacity_ht[kk] = loh_ht * capacity_ht_rated                      #eq13: STORAGE TANK initial condition at t=0
            flag = 1
            
       if flag == 0:
        model.capacity_ht[kk] = loh_ht * capacity_ht_rated                      #eq13: STORAGE TANK initial condition at t=0
        flag = 1
        else:
            model.capacity_ht[kk] = model.capacity_ht[kk - 1]
            # STORAGE BOTTLE
            capacity_prev_bo = capacity_bo_rated
            

'''


