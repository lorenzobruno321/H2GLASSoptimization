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

import numpy as np
import globalPARAMETERS_LB as par
import time
import pyomo.environ as pyo
from pyomo.core import maximize
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
from globalPARAMETERS_LB import*
import matplotlib.pyplot as plt


def new_model_OPTIMIZATION(PV_forecast_dataset, LOAD_dataset, loh_ht):
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
    #        H(k+1) =  H[k] + (etaElz*zONEly[k]*Ts)-(zONFc*Ts/etaFc) I'm not sure of this         |
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
    #=========================================================================|
    #   [1]. CVXPY : https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions-for-non-standard-platforms
    #   [2]. GUROBI: https://www.gurobi.com/documentation/9.0/quickstart_windows/ins_the_anaconda_python_di.html#section:Anaconda
    #====================================================================================================================
    #-------------------- Free Variables -------------------------------------
    #====================================================================================================================
    ## # Definition of SET VECTORS using PYOMO tools
    #==========================================================================
    model = pyo.AbstractModel()
    model.t = pyo.Set(initialize=time_vec)
    flag = 0
    #====================================================================
    ## ELECTROLYSER
    #====================================================================
    model.power_in_ele = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.power_out_ele = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.power_out_ele_bur = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.power_out_ele_cp = pyo.Var(model.t, domain=pyo.Reals, initialize=0)

    model.optDELTA_ele = pyo.Var(model.t, domain=pyo.Binary, initialize=0)
    model.power_OPT_ele = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.optZON_ele = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ====================================================================
    ## PV SUPPLY
    # ====================================================================
    model.power_pv = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.list_pv = pyo.Param(model.t, initialize=PV_forecast_dataset)
    # ====================================================================
    ## FURNACE LOAD
    # ====================================================================
    list_ = dict_load_furnace(import_load_furnace)
    model.list_load_furnace = pyo.Param(model.t, initialize=LOAD_dataset)
    # ====================================================================
    ## GRID SUPPLY
    # ====================================================================
    model.power_grid = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ====================================================================
    ## COMPRESSOR
    # ====================================================================
    model.power_in_cp = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ====================================================================
    ## HYDROGEN STORAGE TANK
    # ====================================================================
    model.power_in_ht = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.power_out_ht = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.capacity_ht = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ====================================================================
    ## HYDROGEN BOTTLE TANK
    # ====================================================================
    model.power_out_bo = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.capacity_bo = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ====================================================================
    ## BURNER
    # ====================================================================
    model.power_in_bur = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    model.power_out_bur = pyo.Var(model.t, domain=pyo.Reals, initialize=0)
    # ===================================================================================================================
    ## # Definition of CONSTRAINTS
    # ===================================================================================================================
    model.constraints = pyo.ConstraintList()

    #kk is cycle over the time horizon (life)
    for kk in range(time_vec):
        if kk > 0:
            #ELECTROLYSER
            power_prev_ele = model.power_out_ele[kk - 1]
            #STORAGE TANK
            capacity_prev_ht = model.capacity_ht[kk - 1]
            # STORAGE BOTTLE
            capacity_prev_bo = model.capacity_bo[kk - 1]

        else:
            # ELECTROLYSER
            power_prev_ele = power_0_ele
            # STORAGE TANK
            if flag == 0:
                capacity_prev_ht = loh_ht * capacity_ht_rated                      #eq13: STORAGE TANK initial condition at t=0
                flag = 1
            else:
                capacity_prev_ht = model.capacity_ht[kk - 1]
            # STORAGE BOTTLE
            capacity_prev_bo = capacity_bo_rated

    # =========================================================================
    ## eq1: PV GENERATION DEFINITION
    # ==========================================================================
    model.constraints.add( model.model.power_pv[kk] <= cap_installed_pv*model.list_pv[kk] )
    # =========================================================================
    ## eq2: POWER BALANCE node1
    # ==========================================================================
    model.constraints.add( model.power_grid[kk] + model.power_pv[kk] == model.power_in_ele[kk] + model.power_in_cp[kk] )
    # =========================================================================
    ## eq3: ELECTROLYSER EFFICIENCY
    # ==========================================================================
    model.constraints.add( model.power_out_ele[kk] == efficiency_ele * model.power_in_ele[kk] )
    # =========================================================================
    ## eq4: LINEARIZATION of the PROBLEM: power_OPT_ele definition
    # ==========================================================================
    model.constraints.add( model.power_OPT_ele[kk] == model.power_in_ele[kk] - model.power_out_ele[kk] )
    # =========================================================================
    ## eq5: LINEARIZATION of the PROBLEM: Z definition
    # ==========================================================================
    model.constraints.add( model.optZON_ele[kk] == model.optDELTA_ele[kk] * model.power_OPT_ele[kk] )
    # =========================================================================
    ## eq6: LINEARIZATION of the PROBLEM: constraint on Z
    # ==========================================================================
    model.constraints.add( - M <= model.optZON_ele[kk] <= M )
    # =========================================================================
    ## eq7: LINEARIZATION of the PROBLEM: constraint on Z - power_OPT_ele
    # ==========================================================================
    model.constraints.add(- M*(1-model.optDELTA_ele[kk]) <= model.optZON_ele[kk] - model.power_OPT_ele[kk] <= M*(1-model.optDELTA_ele[kk]) )
    # =========================================================================
    ## eq8: RAMP UP equation
    # ==========================================================================
    model.constraints.add( abs(model.power_out_ele[kk] - power_prev_ele) <= val_rampup )
    # =========================================================================
    ## eq9: POWER BALANCE node2
    # ==========================================================================
    model.constraints.add( model.power_out_ele[kk] == model.power_out_ele_bur[kk] + model.power_out_ele_cp[kk] )
    # =========================================================================
    ## eq10: POWER required by the COMPRESSOR
    # ==========================================================================
    model.constraints.add( model.power_in_cp[kk] == model.power_out_ele_cp[kk]*compression_work/LHV_h2 )
    # =========================================================================
    ## eq11: COMPRESSOR POWER BALANCE
    # ==========================================================================
    model.constraints.add( model.power_in_cp[kk] + model.power_out_ele_cp[kk] == model.power_in_ht[kk] )
    # =========================================================================
    ## eq12: ENERGY BALANCE in the STORAGE TANK
    # ==========================================================================
    model.constraints.add(model.capacity_ht[kk+1] == model.capacity_ht[k] + model.power_in_ht[kk] - model.power_out_ht[kk])
    # =========================================================================
    ## eq13: STORAGE TANK initial condition --> see line 165
    # ==========================================================================
    # =========================================================================
    ## eq14: ENERGY BALANCE in the STORAGE TANK : CAPACITY CONSTRAINT
    # ==========================================================================
    model.constraints.add( perc_min_ht * capacity_ht_rated <= model.capacity_ht[kk] <= perc_max_ht * capacity_ht_rated )
    # =========================================================================
    ## eq15: ENERGY BALANCE in the STORAGE BOTTLE
    # ==========================================================================
    model.constraints.add( model.capacity_bo[kk+1] == model.capacity_bo[kk]  - model.power_out_bo[kk] )
    # =========================================================================
    ## eq16: STORAGE BOTTLE initial condition --> see line 170
    # ==========================================================================
    # =========================================================================
    ## eq17: POWER BALANCE node3
    # ==========================================================================
    model.constraints.add( model.power_out_ele_bur[kk] + model.power_out_bo[kk] + model.power_out_ht[kk] == model.power_in_bur[kk] )
    # =========================================================================
    ## eq18: BURNER EFFICIENCY
    # ==========================================================================
    model.constraints.add( model.power_out_bur[kk] == efficiency_bur * model.power_in_bur[kk] )
    # =========================================================================
    ## eq19: BURNER BALANCE: constraint on POWER AT INLET
    # ==========================================================================
    model.constraints.add( perc_min_bur*power_bur_rated <= model.power_in_bur[kk] <= perc_max_bur*power_bur_rated )
    # =========================================================================
    ## eq20: POWER BALANCE: LOAD CONSTRAINT
    # ==========================================================================
    model.constraints.add( model.power_out_bur[kk] == model.list_load_furnace[kk])
    # ===================================================================================================================
    ## # COST FUNCTION of ELECTROLYSER AND STORAGES
    # ===================================================================================================================
    def obj_cost(m):
        return sum(m.power_out_ele[kk]*CAPEX_ele for kk in range(time_vec))
    model.obj_cost = pyo.Objective(rule=obj_cost)
    solver = SolverFactory('glpk')
    solver.solve(model, tee=True)
    log_infeasible_constraints(model)
    # ===================================================================================================================
    ## # STORE the values model.optimization
    # ===================================================================================================================
    POWER_OUT_ELE = []
    OPTDELTA_ELE = []
    for ii in range(time_vec):
        POWER_OUT_ELE.append(pyo.value(model.power_out_ele[ii]))
        OPTDELTA_ELE.append(pyo.value(model.optDELTA_ele[ii]))
    return [POWER_OUT_ELE, OPTDELTA_ELE]


updatedp= []


#xtest= np.array(POWER_OUT_ELE)

'''
plt.figure()
plt.plot(np.arange(0,time_vec), list(map(float, np.array(POWER_OUT_ELE))))
plt.grid
plt.show()
'''








#########################################################################################################################
"""
x = [2, 3, 4]
y = [2, 3, 4]
plt.figure()
plt.plot(x, y)
plt.grid
plt.show()


mylist = list_pv_dict.items()
x,y = zip(*mylist)
#x = time_vec
#y = list_pv_dict
plt.figure()
plt.plot(x, y)
plt.grid
plt.show()


mylist = list_pv_dict.items()
x,y = zip(*mylist)
#x = time_vec
#y = list_pv_dict
plt.figure()
plt.grid
plt.plot(x, y)
plt.xlabel("hour [h]")
plt.ylabel("Forecast value [kW]")
plt.show()


mylist = list_load_furnace_dict.items()
x,y = zip(*mylist)
#x = time_vec
#y = list_pv_dict
#plt.figure()
plt.plot(x, y,  marker = "o", color = 'red')
plt.grid
plt.show()


#mylist = list_load_furnace_dict.items()
#x,y = zip(*mylist)
x = time_vec
y = POWER_OUT_ELE
#plt.figure()
plt.plot(time_vec, POWER_OUT_ELE,  marker = "o", color = 'red')
plt.grid
plt.show()


    #return [power_in_ele, power_out_ele, power_out_ele_bur, power_out_ele_cp, optDELTA_ele, power_OPT_ele, optZON_ele, power_pv, power_grid, power_in_cp, power_in_ht, power_out_ht, capacity_ht, power_out_bo, capacity_bo, power_in_bur, power_out_bur]
"""

