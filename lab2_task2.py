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
W_MAX = 1.5         # Maximum wait time of EVs before they renege - assumed

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
    global cust_cnt
    global charge_queue
    global charging_EVs
    global standby_queue
    global cust_queue
    
    EV_cnt_charging = 0
    EV_cnt_in_charge_queue = 0
    EV_cnt_in_standby = 0
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
        self.serviced_custs = []
        
        self.EVarr = 0
        self.EV_serviced = 0
        self.EV_reneged = 0
        
        self.Custarr = 0
        self.Cust_serviced = 0
        self.Cust_unserviced = 0
        
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
        
        data.serviced_EVs.append(client.id)
        data.EV_serviced += 1
        
        data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
        data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
                
        FES.put((time + recharge_time, client_id, "Stand By"))
        
# Function to handle completion of charging of EVs
def EV_charge_completion(time, client_id, FES, charge_queue, standby_queue):
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global EV_cnt_in_standby
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
        
        initial_charge = client.initial_charge
        charging_rate = round(R, 2)
        recharge_time = round((C - initial_charge) / charging_rate, 2)
        
        #print(client.id + ' is charging at ' + str(round(time,2)))
        EV_cnt_in_charge_queue -= 1
        EV_cnt_charging += 1
        charging_EVs.append(client.id)
        
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
                charge_queue.remove([x for x in data.EVs if x.id == client_id][0])
                data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                #print(client_id + ' misses service due to long wait time and leaves.')
                EV_cnt_in_charge_queue -= 1
                
                # Update system performance measurement metrics
                data.EV_reneged += 1
                data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
                
                next
        
        # For other events, call the corresponding event handler function
        if event_type in ["EV_arrival", "Cust_arrival"]:
            arrival(client_id, time, FES, charge_queue, cust_queue)
        elif event_type == "Stand By":
            EV_charge_completion(time, client_id, FES, charge_queue, standby_queue)
        elif event_type in ['Cust_service']:
            customer_service(time, client_id, FES, cust_queue, standby_queue)

"""
Define support functions to help analyze and visualize the performance metrics

"""

# Fishman's method of column means to smoothen the trends
def fishmans_method(avg_waiting_delay_list, colname, ylabel, transient_cutoff_time):
    
    metric = []
    time = []
    for i in range(len(avg_waiting_delay_list)):
        metric.append([x[colname] for x in avg_waiting_delay_list[i]])
        time.append([x['time'] for x in avg_waiting_delay_list[i]])
    
    avg_metric = pd.DataFrame(metric).dropna(axis=1).mean().tolist()
    avg_time = pd.DataFrame(time).dropna(axis=1).mean().tolist()
    
    cutoff_idx = [x for x, val in enumerate(avg_time) if val > transient_cutoff_time][0]
    avg_metric = avg_metric[cutoff_idx:]
    avg_time = [x for x in avg_time[cutoff_idx:]]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(avg_time, avg_metric,'r')
    # plt.title('Fishmans Method: Column Averages')
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.grid(True)
    
    plt.show()
    
    return avg_metric, avg_time

# Generic confidence interval function
def mean_CI(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,4), round(m-h,4), round(m+h,4)

# Truncating transient period and calculating confidence interval of the mean of the performance metric
def confidence_interval(avg_waiting_delay_list, colname, transient_cutoff_time):
    
    metric = []
    time = []
    for i in range(len(avg_waiting_delay_list)):
        metric.append([x[colname] for x in avg_waiting_delay_list[i]])
        time.append([x['time'] for x in avg_waiting_delay_list[i]])
    
    metric_means = []
    for i in range(len(metric)):  
        cutoff_idx = [x for x, val in enumerate(time[i]) if val > transient_cutoff_time][0]
        metric_means.append(sum(metric[i][cutoff_idx:]) / len(metric[i][cutoff_idx:]))
        
    return mean_CI(metric_means)


"""
Analysis of Performance Metrics for the changed inputs

"""

# Define all performance metrics to be tracked
avg_waiting_delay_list = []
missed_EV_service_prob_list = []
missed_cust_service_prob_list = []
EV_waiting_delay_list = []
charge_queue_length_list = []
avg_charge_queue_length_list = []
standby_queue_length_list = []
avg_standby_queue_length_list = []

# Define number of runs and run the simulation
N_runs = 200
for i in range(N_runs):
    
    RANDOM_SEED = int(random.random()*10000)
    
    refresh_initial_params()
    data = Measure()
    main_simulation(RANDOM_SEED)
    
    avg_waiting_delay_list.append(data.EV_avg_waiting_delay)
    missed_EV_service_prob_list.append(data.EV_missed_service_prob)
    missed_cust_service_prob_list.append(data.Cust_missed_service_prob)
    EV_waiting_delay_list.append(data.EV_waiting_delay)
    charge_queue_length_list.append(data.charge_queue_length)
    avg_charge_queue_length_list.append(data.avg_charge_queue_length)
    standby_queue_length_list.append(data.standby_queue_length)
    avg_standby_queue_length_list.append(data.avg_standby_queue_length)

# Defining truncation time for the new inputs
transient_cutoff_time = 72

# Plot the performance metrics for visualization

smoothed_metric, smoothed_time = fishmans_method(EV_waiting_delay_list, 
                                                 'wait_delay', 'Wait Time', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(avg_waiting_delay_list, 
#                                                 'avg_wait_delay', 'Average Wait Time', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(missed_EV_service_prob_list, 
#                                                 'prob', 'Missed Service Probability - EV', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(missed_cust_service_prob_list, 
#                                                 'prob', 'Missed Service Probability - Customer', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(missed_cust_service_prob_list, 
#                                                 'prob', 'Missed Service Probability - Customer', transient_cutoff_time)
smoothed_metric, smoothed_time = fishmans_method(charge_queue_length_list, 
                                                 'qlength', 'Charge Queue Length', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(avg_charge_queue_length_list, 
#                                                 'avg_qlength', 'Average Charge Queue Length', transient_cutoff_time)
smoothed_metric, smoothed_time = fishmans_method(standby_queue_length_list, 
                                                 'qlength', 'Standy Queue Length', transient_cutoff_time)
#smoothed_metric, smoothed_time = fishmans_method(avg_standby_queue_length_list, 
#                                                 'avg_qlength', 'Average Standy Queue Length', transient_cutoff_time)

# Get the confidence interval of the performance metrics

print('Average Waiting Delay: ' + str(confidence_interval(avg_waiting_delay_list, 'avg_wait_delay',transient_cutoff_time)))
print('Missed Service Probability - EV: ' + str(confidence_interval(missed_EV_service_prob_list, 'prob', transient_cutoff_time)))
print('Missed Service Probability - Customer: ' + str(confidence_interval(missed_cust_service_prob_list, 'prob', transient_cutoff_time)))
print('Average Charge Queue Length: ' + str(confidence_interval(avg_charge_queue_length_list, 'avg_qlength', transient_cutoff_time)))
print('Average Standby Queue Length: ' + str(confidence_interval(avg_standby_queue_length_list, 'avg_qlength', transient_cutoff_time)))