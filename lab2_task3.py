"""
Objective: Simulate a single EV charging station and analyze system performance

"""

# Import libraries
import random
from queue import PriorityQueue
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect

# Set working directory
fn = inspect.getframeinfo(inspect.currentframe()).filename
os.chdir(os.path.dirname(os.path.abspath(fn)))

"""
Define inputs and key characteristics of the system

"""

C = 20              # Battery capacity of each EV - 20 kWh given
I_C = [0, 4]        # Max and Min of Initial Charge of each EV - assumed
R = 7.5             # Rate of Charging at the SCS - 5 kW to 10 kW given
M_C = 20            # Minimum Charge needed for EV to be picked up - given
N_SCS = 2          # Number of Charging Sockets at the Station - assumed
area = 'RESIDENTIAL'# Area of the location of the SCS
W_MAX = 5         # Maximum wait time of EVs before they renege - assumed
postpone_hours = [x for x in range(8,12)] + [x for x in range(16,23)] # Postponing charging of EVs during this period

"""
Assumptions of EV and Customer arrival rates depending on time of day and area of SCS

"""

if area == 'RESIDENTIAL':

    # 0 to 7 - (low, low)
    # 7 to 12 - (low, high)
    # 12 to 18 - (med, med)
    # 18 to 22 - (high, low)
    # 22 to 23 - (med, low)
    EV_arr_rates = [1,2,5]
    cust_arr_rates = [1,1.5,3]
    
    INTER_T = [EV_arr_rates[0]]*12 + [EV_arr_rates[1]]*6 + [EV_arr_rates[2]]*4 + [EV_arr_rates[1]]*2
    CUST_T = [cust_arr_rates[0]]*7 + [cust_arr_rates[2]]*5 + [cust_arr_rates[1]]*6 + [cust_arr_rates[0]]*6

elif area == 'BUSINESS':
    
    # 0 to 7 - (low, low)
    # 7 to 12 - (high, low)
    # 12 to 18 - (med, med)
    # 18 to 22 - (med, high)
    # 22 to 23 - (med, med)
    EV_arr_rates = [1,2,10]
    cust_arr_rates = [1,1.5,7]
    
    INTER_T = [EV_arr_rates[0]]*7 + [EV_arr_rates[2]]*5 + [EV_arr_rates[1]]*12
    CUST_T = [cust_arr_rates[0]]*12 + [cust_arr_rates[1]]*6 + [cust_arr_rates[2]]*4 + [cust_arr_rates[1]]*2

"""
Set and Reset Initial Simulation Parameters

"""

SIM_TIME = 24*7

# Counts of EVs and Customers in different queues
EV_cnt_charging = 0
EV_cnt_in_charge_queue = 0
EV_cnt_in_standby = 0
EV_cnt_postponed = 0
cust_cnt = 0

# Details of EVs and Customers in different queues for tracking purpose
charge_queue = []
charging_EVs = []
standby_queue = []
cust_queue = []

# Function to reset the initial parameters for different runs of the  simulation
def refresh_initial_params():
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global EV_cnt_in_standby
    global EV_cnt_postponed
    global cust_cnt
    global charge_queue
    global charging_EVs
    global standby_queue
    global cust_queue
    
    EV_cnt_charging = 0
    EV_cnt_in_charge_queue = 0
    EV_cnt_in_standby = 0
    EV_cnt_postponed = 0
    cust_cnt = 0
    
    charge_queue = []
    charging_EVs = []
    standby_queue = []
    cust_queue = []

"""
Classes to Define EV objects, Client objects and Measurement objects

"""

# EVs defined by their arrival time, initial charge and expected renege time
class EV_Client:
    def __init__(self, client_id, arrival_time):
        self.id = client_id
        self.arrival_time = arrival_time
        self.initial_charge = round(random.uniform(*I_C), 2)
        self.renege_time = self.arrival_time + W_MAX*1.0001

# Customers defined by their arrival time        
class Cust_Client:
    def __init__(self, client_id, arrival_time):
        self.id = client_id
        self.arrival_time = arrival_time

# Measure objects contain key system performance metrics tracked over simulation time
class Measure:
    def __init__(self):
        
        self.EVs = []
        self.Customers = []
        
        self.serviced_EVs = []
        self.reneged_EVs = []
        self.serviced_custs = []
        
        self.EVarr = 0
        self.EV_serviced = 0
        self.EV_reneged = 0
        
        self.Custarr = 0
        self.Cust_serviced = 0
        self.Cust_unserviced = 0
        
        self.recharge_start_time = {}
        self.recharge_end_time = {}
        
        self.EV_waiting_delay = []
        self.EV_avg_waiting_delay = []
        
        self.EV_missed_service_prob = []
        self.Cust_missed_service_prob = []
        
        self.charge_queue_length = []
        self.avg_charge_queue_length = []
        
        self.standby_queue_length = []
        self.avg_standby_queue_length = []

"""
Define functions to handle events in the system

"""

#Function to handle arrival of clients - both EVs and Customers

def arrival(client_id, time, FES, charge_queue, cust_queue):
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global EV_cnt_postponed
    global cust_cnt
    global data
    
    #print(client_id + ' arrived at ' + str(time))
    
    if client_id[0] == 'E':
        data.EVarr += 1
        data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
        inter_arrival = random.expovariate(INTER_T[int(time % 24)])
        FES.put((time + inter_arrival, client_id[0] + str(int(client_id[1:]) + 1), "EV_arrival"))
        EV_cnt_in_charge_queue += 1
        client = EV_Client(client_id, time)
        data.EVs.append(client)
        charge_queue.append(client)
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        FES.put((client.renege_time, client.id, "Renege"))
    elif client_id[0] == 'C':
        data.Custarr += 1
        data.Cust_missed_service_prob.append({'time': time, 'prob': round(data.Cust_unserviced / data.Custarr, 2)})
        inter_arrival = random.expovariate(CUST_T[int(time % 24)])
        FES.put((time, client_id, "Cust_service"))
        FES.put((time + inter_arrival, client_id[0] + str(int(client_id[1:]) + 1), "Cust_arrival"))
        cust_cnt += 1
        client = Cust_Client(client_id, time)
        data.Customers.append(client)
        cust_queue.append(client)

    if client_id[0] == 'E' and EV_cnt_charging < N_SCS:
        
        if random.choices([1,0], [f[int(time % 24)], 1-f[int(time % 24)]])[0] == 1:
            FES.put((time + random.uniform(0,T_max[int(time % 24)]), client.id, 'Postpone'))
            #print('Charging of ' + client.id + ' is postponed')
            EV_cnt_in_charge_queue -= 1
            EV_cnt_postponed += 1
            
            charge_queue.remove(client)
        
        else:
            initial_charge = client.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            
            client = charge_queue.pop(0)
            data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
            data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
            #print(client.id + ' is charging at ' + str(time))
            EV_cnt_in_charge_queue -= 1
            EV_cnt_charging += 1
            
            charging_EVs.append(client.id)
            
            data.recharge_start_time[client.id] = time
            data.recharge_end_time[client.id] = time + recharge_time
            
            data.serviced_EVs.append(client.id)
            data.EV_serviced += 1
            
            data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
            data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
                    
            FES.put((time + recharge_time, client.id, "Stand By"))
        
# Function to handle completion of charging of EVs
def EV_charge_completion(time, client_id, FES, charge_queue, standby_queue):
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global EV_cnt_in_standby
    global EV_cnt_postponed
    global data
    
    client = [x for x in data.EVs if x.id == client_id][0]
    standby_queue.append(client)
    data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
    data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
        
    EV_cnt_charging -= 1
    EV_cnt_in_standby += 1
    
    charging_EVs.remove(client.id)
    
    if EV_cnt_in_charge_queue > 0:
        client = charge_queue.pop(0)
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        if random.choices([1,0], [f[int(time % 24)], 1-f[int(time % 24)]])[0] == 1:
            FES.put((time + random.uniform(0,T_max[int(time % 24)]), client.id, 'Postpone'))
            #print('Charging of ' + client.id + ' is postponed')
            EV_cnt_in_charge_queue -= 1
            EV_cnt_postponed += 1
        
        else:
            
            initial_charge = client.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            
            #print(client.id + ' is charging at ' + str(round(time,2)))
            EV_cnt_in_charge_queue -= 1
            EV_cnt_charging += 1
            charging_EVs.append(client.id)
            
            data.recharge_start_time[client.id] = time
            data.recharge_end_time[client.id] = time + recharge_time
            
            data.serviced_EVs.append(client.id)
            data.EV_serviced += 1
            
            data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
            data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
                    
            FES.put((time + recharge_time, client.id, "Stand By"))

# Function to handle servicing of arriving customers
def customer_service(time, client_id, FES, cust_queue, standby_queue):
    global cust_cnt
    global EV_cnt_in_standby
    global data
        
    client_cust = cust_queue.pop(0)
    
    if len(standby_queue) > 0:
    
        client_EV = standby_queue.pop(0)
        #print(client_cust.id + ' takes ' + client_EV.id + ' and leaves at ' + str(time))
        data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
        data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})

        
        cust_cnt -= 1
        EV_cnt_in_standby -= 1
        
        data.serviced_custs.append(client_cust.id)
        data.Cust_serviced += 1
        
    else:
        #print(client_cust.id + ' misses service and leaves.')
        cust_cnt -= 1
        data.Cust_unserviced += 1
        data.Cust_missed_service_prob.append({'time': time, 'prob': round(data.Cust_unserviced / data.Custarr, 2)})

def EV_postponed_rejoin(time, client_id, FES, charge_queue, charging_EVs):
    
    global EV_cnt_in_charge_queue
    global EV_cnt_postponed
    global EV_cnt_charging
    global data
    
    client = [x for x in data.EVs if x.id == client_id][0]
    
    if client_id not in data.reneged_EVs:
        
        EV_cnt_in_charge_queue += 1
        EV_cnt_postponed -= 1
        
        charge_queue.append(client)
        
        # Starting charging of the EV if there are available sockets
        if EV_cnt_charging < N_SCS:
        
            initial_charge = client.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            
            client = charge_queue.pop(0)
            data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
            data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
            #print(client.id + ' is charging at ' + str(time))
            EV_cnt_in_charge_queue -= 1
            EV_cnt_charging += 1
            
            charging_EVs.append(client.id)
            
            data.recharge_start_time[client.id] = time
            data.recharge_end_time[client.id] = time + recharge_time
            
            data.serviced_EVs.append(client.id)
            data.EV_serviced += 1
            
            data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
            data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
                    
            FES.put((time + recharge_time, client.id, "Stand By"))

"""
Retrieve Electricity Price data

"""

def get_electricity_prices(filename='electricity_prices.csv'):
    
    prices = pd.read_csv(filename, header=None)
    
    spring_prices = prices.iloc[:,[1,2,3]]
    spring_prices.columns = ['Hour', 'Season', 'Price']
    
    summer_prices = prices.iloc[:,[1,4,5]]
    summer_prices.columns = ['Hour', 'Season', 'Price']
    
    fall_prices = prices.iloc[:,[1,6,7]]
    fall_prices.columns = ['Hour', 'Season', 'Price']
    
    winter_prices = prices.iloc[:,[1,8,9]]
    winter_prices.columns = ['Hour', 'Season', 'Price']

    electricity_prices = spring_prices.append([summer_prices, fall_prices, winter_prices]).reset_index(drop=True)
    electricity_prices['Season'] = electricity_prices['Season'].apply(lambda x: x.replace(":",""))
    
    return electricity_prices

"""
Run a simulation from time = 0 to SIM_TIME

"""

def main_simulation(RANDOM_SEED):
    
    global EV_cnt_in_charge_queue
    global data
    
    data = Measure()

    # Set seed
    random.seed(RANDOM_SEED)
    
    # Define event queue        
    FES = PriorityQueue()

    # Initialize starting events
    time = 0    
    FES.put((random.expovariate(INTER_T[0]), 'E0', "EV_arrival"))
    FES.put((random.expovariate(CUST_T[0]), 'C0', "Cust_arrival"))
    
    # Simulate until defined simulation time
    while time < SIM_TIME:
        
        # Get the immediate next scheduled event
        (time, client_id, event_type) = FES.get()      
        
        # If the event is an EV renege event, 
        if event_type == "Renege":
            
            if client_id in data.serviced_EVs:
                (time, client_id, event_type) = FES.get()
            
            else:
                
                # Update relevant simulation tracking parameters
                if [x for x in data.EVs if x.id == client_id][0] in charge_queue:
                
                    charge_queue.remove([x for x in data.EVs if x.id == client_id][0])
                    data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                    data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                    #print(client_id + ' misses service due to long wait time and leaves.')
                    EV_cnt_in_charge_queue -= 1
                
                # Update system performance measurement metrics
                data.EV_reneged += 1
                data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
                
                data.reneged_EVs.append(client_id)
                
                next
        
        # For other events, call the corresponding event handler function
        if event_type in ["EV_arrival", "Cust_arrival"]:
            arrival(client_id, time, FES, charge_queue, cust_queue)
        elif event_type == "Stand By":
            EV_charge_completion(time, client_id, FES, charge_queue, standby_queue)
        elif event_type in ['Cust_service']:
            customer_service(time, client_id, FES, cust_queue, standby_queue)
        elif event_type == ['Postpone']:
            EV_postponed_rejoin(time, client_id, FES, charge_queue, charging_EVs)


"""
Analysis of Performance Metrics for the new scenario

"""

# Plot utilization by hour
def plot_utilization(util_by_hour):
    
    avg_metric = pd.DataFrame(util_by_hour).mean().tolist()
    avg_time = [x for x in range(24)]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(avg_time, avg_metric)
    plt.title('Avg. BSS Utilization by Hour')
    plt.xlabel('Hour of day')
    plt.ylabel('Utilization %')
    
    fig.show()
    
""" 
Solution to first part of the question
"""

# Define f and Tmax
f_list = [[y*(x in postpone_hours) for x in range(24)] for y in [0.1, 0.2, 0.3, 0.4, 0.5]]
T_max_list = [[y*(x in postpone_hours) for x in range(24)] for y in [1, 2, 3, 4, 5, 6, 7, 8]]

# Get electricity prices
electricity_prices = get_electricity_prices()    
prices_summer = electricity_prices.loc[electricity_prices['Season'] == 'SUMMER','Price'].tolist()
prices_winter = electricity_prices.loc[electricity_prices['Season'] == 'WINTER','Price'].tolist()

# Run 200 simulations and get average cost per EV and number of EVs services
cost_summer_list = []
cost_winter_list = []
serviced_EVs_cnt = []
util_list = []
EV_missed_service_prob = []
for f in f_list:
    
    temp_list_summer = []
    temp_list_winter = []
    temp_list_serviced_EVs = []
    temp_list_util = []
    for T_max in T_max_list:
        
        util_by_hour = []
        cost_summer = []
        cost_winter = []
        EVs_serviced_cnt = []
        
        for i in range(200):
            
            RANDOM_SEED = int(random.random()*10000)
                
            refresh_initial_params()
            data = Measure()
            main_simulation(RANDOM_SEED)
            
            temp_util_list = [0]*SIM_TIME
            for EV in data.serviced_EVs:
                recharge_start_time = data.recharge_start_time[EV]
                recharge_end_time = data.recharge_end_time[EV]
                if recharge_end_time > SIM_TIME:
                    recharge_time = SIM_TIME - recharge_start_time
                else:
                    recharge_time = recharge_end_time - recharge_start_time
                
                temp_time = (recharge_start_time - int(recharge_start_time)) + recharge_time
                temp_time_list = [1 for x in range(int(temp_time))] + [round(temp_time - int(temp_time),2)]
                temp_time_list[0] = round(temp_time_list[0] - (recharge_start_time - int(recharge_start_time)),2)
                for j in range(int(recharge_start_time), min(int(recharge_start_time) + len(temp_time_list), SIM_TIME)):
                    temp_util_list[j] = temp_util_list[j] + temp_time_list[j - int(recharge_start_time)]
                    
            temp_util_list_24 = []
            for j in range(24):
                temp_util_list_24.append(sum([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]) / len([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]))
                
            util_by_hour.append([x/N_SCS for x in temp_util_list_24])
            
            EVs_serviced_cnt.append(len(data.serviced_EVs))
            
        avg_util_by_hour = pd.DataFrame(util_by_hour).mean().tolist()
        avg_EVs_serviced = sum(EVs_serviced_cnt) / len(EVs_serviced_cnt) / 7
        
        cost_summer = [x*R*N_SCS*y/1000 for x, y in zip(avg_util_by_hour, prices_summer)]
        cost_winter = [x*R*N_SCS*y/1000 for x, y in zip(avg_util_by_hour, prices_winter)]
            
        cost_summer = sum(cost_summer)
        cost_winter = sum(cost_winter)
        
        cost_per_EV_summer = cost_summer / avg_EVs_serviced
        cost_per_EV_winter = cost_winter / avg_EVs_serviced
            
        temp_list_summer.append(cost_per_EV_summer)
        temp_list_winter.append(cost_per_EV_winter)
        temp_list_serviced_EVs.append(avg_EVs_serviced)
            
        print('f: ' + str(f) + ', ' + 'T_max: ' + str(T_max))
            
    cost_summer_list.append(temp_list_summer)
    cost_winter_list.append(temp_list_winter)
    serviced_EVs_cnt.append(temp_list_serviced_EVs)
    util_list.append(avg_util_by_hour)

# Save the results to dataframes
cost_summer_df = pd.DataFrame(cost_summer_list)
cost_winter_df = pd.DataFrame(cost_winter_list)
serviced_EVs_cnt_df = pd.DataFrame(serviced_EVs_cnt)

""" 
Solution to second part of the question
"""

# Define f and Tmax
f = [0]*8 + [0.2]*5 + [0.1]*3 + [0.3]*8
T_max = [0]*8 + [3]*5 + [1]*3 + [4]*8
 
# Get electricity prices
electricity_prices = get_electricity_prices()    
prices_summer = electricity_prices.loc[electricity_prices['Season'] == 'SUMMER','Price'].tolist()
prices_winter = electricity_prices.loc[electricity_prices['Season'] == 'WINTER','Price'].tolist()

# Run 200 simulations and get average cost per EV and number of EVs services
util_by_hour = []
EVs_serviced_cnt = []
for i in range(200):
    
    RANDOM_SEED = int(random.random()*10000)
        
    refresh_initial_params()
    data = Measure()
    main_simulation(RANDOM_SEED)
    
    temp_util_list = [0]*SIM_TIME
    for EV in data.serviced_EVs:
        recharge_start_time = data.recharge_start_time[EV]
        recharge_end_time = data.recharge_end_time[EV]
        if recharge_end_time > SIM_TIME:
            recharge_time = SIM_TIME - recharge_start_time
        else:
            recharge_time = recharge_end_time - recharge_start_time
        
        temp_time = (recharge_start_time - int(recharge_start_time)) + recharge_time
        temp_time_list = [1 for x in range(int(temp_time))] + [round(temp_time - int(temp_time),2)]
        temp_time_list[0] = round(temp_time_list[0] - (recharge_start_time - int(recharge_start_time)),2)
        for j in range(int(recharge_start_time), min(int(recharge_start_time) + len(temp_time_list), SIM_TIME)):
            temp_util_list[j] = temp_util_list[j] + temp_time_list[j - int(recharge_start_time)]
            
    temp_util_list_24 = []
    for j in range(24):
        temp_util_list_24.append(sum([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]) / len([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]))
        
    util_by_hour.append([x/N_SCS for x in temp_util_list_24])
    
    EVs_serviced_cnt.append(len(data.serviced_EVs))
            
avg_util_by_hour = pd.DataFrame(util_by_hour).mean().tolist()
avg_EVs_serviced = sum(EVs_serviced_cnt) / len(EVs_serviced_cnt) / 7

cost_summer = [x*R*N_SCS*y/1000 for x, y in zip(avg_util_by_hour, prices_summer)]
cost_winter = [x*R*N_SCS*y/1000 for x, y in zip(avg_util_by_hour, prices_winter)]
    
cost_summer = sum(cost_summer)
cost_winter = sum(cost_winter)

cost_per_EV_summer = cost_summer / avg_EVs_serviced
cost_per_EV_winter = cost_winter / avg_EVs_serviced
            


    


