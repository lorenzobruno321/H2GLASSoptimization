# coding
"""

Created on Mon 10/07/2023 12:28
## Integrated algorithm
#====================================================================
## #[Design]
#====================================================================
# @Coded by     : Lorenzo Bruno
# @email        : lorenzo.bruno@studenti.polito.it
# @Date         : July, 2023
# @Dynamic Model
# @Paper        : UPC collaboration
#====================================================================
## #[Steps of successfully run]
#====================================================================
#   [1]. Install PyCharm as script editor
#   [2]. Install CVXPY toolbox with the independences: OSQP, ECOS, SCS, NumPy, SciPy, Panda
#   [3]. Install MILP solver: GUROBI solver
#   [4]. Install important packages: gurobipy and argparse
#   [5]. Run the file with name "runNewModelMPC"
#   [6].
#   [7].
#====================================================================
## #[Prerequites]
#====================================================================
#
#
#====================================================================
## #[Global Parameters]
#====================================================================

#====================================================================
## TIME DATA for the code
#====================================================================/
time_vec                    vector with all the hours in the year    /
time_step                   time step of 1 hour                      /
life                        time horizont of the plant               /
#====================================================================/
## CHEMICAL PROPERTIES
#====================================================================/
LHV                         Lower Heating Value for hydrogen         /
h2_density                  density of the hydrogen                  /
water_density               density of water                         /
ECI                         Emission Carbon Intensity                /
#====================================================================/
## ELECTROLYSER
#====================================================================/
flow_rate_m3                Flow rate produced by the electrolyser   /
flow_rate                   Flow rate in [kg/s]                      /
efficiency_ele              efficiency of electrolyser               /
val_rampup_data             limiti to the derivative                 /
val_rampup                  limit in terms [kW]                      /
perc_max_ele                upper range of production of h2          /
perc_min_ele                lower range of production of h2          /
CAPEX_ele                   CAPEX of electrolyser                    /
OPEX_ele                    OPEX of electrolyser                     /
INSTALL_ele                 installation costs                       /
REPLACE_ele                 replacement costs                        /
#====================================================================/
## power load at the PV SUPPLY
#====================================================================/
import_PV_supply            import from Excel                        /
list_index                  list of index                            /
def dict_Forecast(xx)       function to build the dictionary         /
list_pv = dict_Forecast(import_PV_supply)
CAPEX_pv_USD                CAPEX of PV                              /
CAPEX_pv                    CAPEX of PV in [€/kWe/year]              /
OPEX_pv                     OPEX of PV in [€/kWe/year]               /
#====================================================================
## GRID
#====================================================================/
cost_energy_grid            cost of the energy in Spain              /
#====================================================================/
## power load at the FURNACE
#====================================================================/
import_load_furnace         import file from .mat file               /
#====================================================================/
## HYDROGEN COMPRESSOR
#====================================================================/
specific_work_cp            specific work in [MJ/kgH2]               /
compression_work            specific work in [kWh]                   /
CAPEX_cp                    CAPEX of compressor                      /
OPEX_cp_USD                 OPEX in [USD/kW]                         /
OPEX_cp                     OPEX in [€/kWe]                          /
#====================================================================/
## HYDROGEN STORAGE TANK
#====================================================================/
loh_ht                      level of hydrogen                        /
CAPEX_ht                    CAPEX of h2 tank                         /
OPEX_ht                     OPEX of h2 tank                          /
#====================================================================/
## HYDROGEN BOTTLE TANK
#====================================================================/
capacity_volume_bo          volume in [liters]                       /
capacity_rated_bo           capacity in [kWh]                        /
CAPEX_bo                    CAPEX of bottle                          /
OPEX_bo                     OPEX of the bottle tank                  /
#====================================================================/
## BURNER
#====================================================================/
efficiency_bur              efficiency a the burner                  /
perc_max_bur                max percentual of production             /
perc_min_bur                min percentual of production             /
CAPEX_bur                   CAPEX of burner                          /
OPEX_bur                    OPEX of burner                           /
"""
import pandas
import pyomo as pyo
from scipy.io import loadmat
#====================================================================
## TIME DATA for the code
#====================================================================
time_end = 8760                                                                         # [h]
#time_step = time_vec[1] - time_vec[0]                                                   # [h]
life = 20                                                                               # [years] lifetime of the plant
#====================================================================
## CHEMICAL PROPERTIES
#====================================================================
LHV_h2 = 33.33                                                                             # [kWh/kg]
density_h2 = 0.0899                                                                     # [kg/m3]
density_h2o = 1000                                                                    # [kg/m3]
ECI = 166                                                                               # [gCO2/kWh] https://www.statista.com/statistics/1290486/carbon-intensity-power-sector-spain/#:~:text=In%202021%2C%20Spain's%20power%20sector,%2FKWh)%20of%20electricity%20generated.
#====================================================================
## ELECTROLYSER
#====================================================================
flow_rate_m3 = 500                                                                      # [Nm3/h] (500 said Eduardo) but 420 from https://www.h-tec.com/fileadmin/user_upload/produkte/produktseiten/HCS/spec-sheet/H-TEC-Datenblatt-HCS-EN-23-03.pdf
flow_rate = flow_rate_m3 / 3600 * density_h2                                            # [kg/s]
efficiency_ele = 0.65                                                                   # [-] H-TEC SYSTEMS PEM Electrolyzer: Hydrogen Cube System and H2GLASSefficiency_ele = 0.65
val_rampup_data = 60                                                                    # [MW/min]
val_rampup = val_rampup_data*1000*60                                                    # [kW] = [MW/min] * [kW/MW] * [min/hour]
perc_max_ele = 1                                                                        # [-] Marocco Gandiglio
perc_min_ele = 0.1                                                                      # [-] Marocco Gandiglio
CAPEX_ele = 1188                                                                        # [€/kWe/year]  https://www.iea.org/reports/electrolysers + Marocco Gandiglio
OPEX_ele = 15.84                                                                        # [€/kWe/year]  Marocco Gandiglio
INSTALL_ele = CAPEX_ele*0.1                                                             # [€/kWe/year]  Marocco Gandiglio
REPLACE_ele = CAPEX_ele*0.35                                                            # [€/kWe/year]  Marocco Gandiglio
M = 1e6
power_0_ele = 0                                                                         # [kW] initial value for the eq8
#====================================================================
## power load at the PV SUPPLY
#====================================================================
import scipy.io as sio



#mat= sio.loadmat('Data_set.mat')
#PV= mat['PV_data']
#PL= mat['PL_data']

import_PV_supply = pandas.read_excel("pv_supply_barcelona_1kwp.xlsx", sheet_name='pv_supply_barcelona_1kwp', header=None, index_col=None)
list_index = list(range(1,import_PV_supply.shape[1]))
def dict_Forecast(xx):
    dict_Forecast_in = {t: xx.iloc[4+t, 2] for t in range(0,time_end)}
    return dict_Forecast_in
list_pv_dict = dict_Forecast(import_PV_supply)
CAPEX_pv_USD = 0.61                                                                     # [USD/We/year]  https://www.statista.com/statistics/971982/solar-pv-capex-worldwide-utility-scale/
CAPEX_pv = CAPEX_pv_USD / 0.92 * 1000                                                   # [€/kWe/year] = [USD/W] * [€/USD] * [W/kW]  https://www.statista.com/statistics/971982/solar-pv-capex-worldwide-utility-scale/
OPEX_pv = OPEX_ele*0.05                                                                 # [€/kWe/year]  https://it.scribd.com/document/514697464/COSTOS-DETALLADO-CAPEX-2019-PLANTA-CALLAO
cap_installed_pv = 1                                                                    # [-] related to the forecast value

mat= sio.loadmat('PV.mat')
PV = mat['PV']
#====================================================================
## GRID
#====================================================================
cost_energy_grid = 0.2966                                                               # [€/kWh*h] https://electricityinspain.com/electricity-prices-in-spain/
#====================================================================
## POWER LOAD at the FURNACE
#====================================================================
#import_load_furnace = pandas.read_excel("thermalload_momo_new.xlsx", sheet_name='foglio1', header=None, index_col=None)
#def dict_load_furnace(xx):
#    dict_load_furnace_in = {t: xx.iloc[0+t, 0] for t in range(0,time_end)}
#    return dict_load_furnace_in
#list_load_furnace_dict = dict_load_furnace(import_load_furnace)

mat= sio.loadmat('thermalload_momo_new.mat')
mat= sio.loadmat('PL.mat')
PL = mat['PL']
#====================================================================
## HYDROGEN COMPRESSOR
#====================================================================
specific_work_cp = 4                                                                    # [MJ/kgH2]
compression_work = specific_work_cp * 1000 / 3600 * flow_rate * 3600                    # [kWh] = [MJ/kgH2] * [kJ/MJ] * [kWh/kJ] * [kgH2/s] * [s]
CAPEX_cp = 1600                                                                         # [€/kWe/year]  Marocco Gandiglio
OPEX_cp_USD = 19                                                                        # [USD/kW] https://emp.lbl.gov/publications/benchmarking-utility-scale-pv
OPEX_cp = OPEX_cp_USD / 0.92                                                            # [€/kWe]
#====================================================================
## HYDROGEN STORAGE TANK
#====================================================================
loh_ht = 0.75
perc_max_ht = 0.9                                                                       # [-] [MOMO]
perc_min_ht = 0.1                                                                       # [-] [MOMO]
capacity_ht_rated = 11.2e3                                                                   # [kWh]             A CASOO     https://core.ac.uk/download/pdf/11653831.pdf
CAPEX_ht = 470                                                                          # [€/kgH2/year]  Marocco Gandiglio
OPEX_ht = OPEX_ele*0.02                                                                 # [€/kgH2/year]  Marocco Gandiglio
#====================================================================
## HYDROGEN BOTTLE TANK
#====================================================================
capacity_volume_bo = 850                                                                # [liters] https://www.mahytec.com/wp-content/uploads/2021/03/CL-DS10-Data-sheet-60bar-850L-EN.pdf
capacity_bo_rated = capacity_volume_bo / 1000 * density_h2 * LHV_h2                        # [kWh] = [litri] * [m3/l] * [kg/m3] * [kWh/kg]
CAPEX_bo = 470/2                                                                        # [€/kgH2/year] like the storage: MOMO said to reduce it so it is dividd by two
OPEX_bo = OPEX_ele*0.02                                                                 # [€/kgH2/year] like the storage:
#====================================================================
## BURNER
#====================================================================
efficiency_bur = 0.95                                                                   # [-] Marocco Gandiglio
perc_max_bur = 1                                                                        # [-] Marocco Gandiglio
perc_min_bur = 0                                                                        # [-] Marocco Gandiglio
power_bur_rated = 1000                                                                  # [kW] a caso (?)
CAPEX_bur = 63.32                                                                       # [€/kWth/year]  Marocco Gandiglio
OPEX_bur = CAPEX_bur*0.05                                                               # [€/kWth/year]  Marocco Gandiglio
