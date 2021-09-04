"""
Objective: Simulate a single EV charging station and analyze system performance

"""

# Import libraries
import random
from queue import PriorityQueue
import pandas as pd
from scipy.stats import ttest_ind
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
import warnings

# to ignore one specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
N_SCS = 2           # Number of Charging Sockets at the Station - assumed

INTER_T = 0.75      # Inter Arrival Time of EVs - constant assumed
CUST_T = 1          # INter Arrival Time of Customers - constant assumed
W_MAX = 1.5         # Maximum wait time of EVs before they renege - assumed

SIM_TIME = 100     # Number of Hours that will be sequentially Simulated

"""
Set and Reset Initial Simulation Parameters

"""
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
        
        #self.avg_energy_cost_per_EV = []
        
        self.charge_queue_length = []
        self.avg_charge_queue_length = []
        
        self.standby_queue_length = []
        self.avg_standby_queue_length = []

"""
Define functions to handle events in the system

"""
# Function to handle arrival of clients - both EVs and Customers
def arrival(client_id, time, FES, charge_queue, cust_queue):
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global cust_cnt
    global data
    
    #print(client_id + ' arrived at ' + str(time))
    
    # If EV arrives
    if client_id[0] == 'E':
        
        client = EV_Client(client_id, time)
        
        # Update relevant counts and parameters for tracking
        data.EVarr += 1
        EV_cnt_in_charge_queue += 1
        
        data.EVs.append(client)
        charge_queue.append(client)
        
        # Update system performance measurement metrics
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
        
        # Schedule next arrival of EV
        inter_arrival = INTER_T
        FES.put((time + inter_arrival, client_id[0] + str(int(client_id[1:]) + 1), "EV_arrival"))
        
        # Schedule renege of the EV
        FES.put((client.renege_time, client.id, "Renege"))
     
    # If Customer arrives
    elif client_id[0] == 'C':
        
        client = Cust_Client(client_id, time)
        
        # Update relevant counts and parameters for tracking
        data.Custarr += 1        
        cust_cnt += 1

        data.Customers.append(client)
        cust_queue.append(client)
        
        # Schedule servicing of the arrived customer
        FES.put((time, client_id, "Cust_service"))

        # Schedule arrival of next customer
        inter_arrival = CUST_T
        FES.put((time + inter_arrival, client_id[0] + str(int(client_id[1:]) + 1), "Cust_arrival"))
    
    # If there are sockets available when an EV arrives, start its charging process
    if client_id[0] == 'E' and EV_cnt_charging < N_SCS:
        
        # Determine recharge time
        initial_charge = client.initial_charge
        charging_rate = round(R, 2)
        recharge_time = round((C - initial_charge) / charging_rate, 2)
        
        # Update paramters used to track the simulation
        client = charge_queue.pop(0)
        charging_EVs.append(client.id)
        #print(client.id + ' is charging at ' + str(time))
        
        EV_cnt_in_charge_queue -= 1
        EV_cnt_charging += 1
        
        # Update system performance measurement metrics
        
        data.serviced_EVs.append(client.id)
        data.EV_serviced += 1
        
        data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
        data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
        
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        # Schedule completion of charging process
        FES.put((time + recharge_time, client_id, "Stand By"))
        
# Function to handle completion of charging of EVs
def EV_charge_completion(time, client_id, FES, charge_queue, standby_queue):
    global EV_cnt_charging
    global EV_cnt_in_charge_queue
    global EV_cnt_in_standby
    global data
    
    client = [x for x in data.EVs if x.id == client_id][0]
    
    # Update relevant counts and parameters for tracking
    standby_queue.append(client)
    charging_EVs.remove(client.id)
    
    #print(client.id + ' charge completed at ' + str(round(time,2)))
    EV_cnt_charging -= 1
    EV_cnt_in_standby += 1
    
    # Update performance metrics
    data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
    data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
    
    # Inititate the charging process of next EV in queue
    if EV_cnt_in_charge_queue > 0:
        
        client = charge_queue.pop(0)
        
        # Determine recharge time
        initial_charge = client.initial_charge
        charging_rate = round(R, 2)
        recharge_time = round((C - initial_charge) / charging_rate, 2)
        
        # Update paramters used to track the simulation
        charging_EVs.append(client.id)
        #print(client.id + ' is charging at ' + str(round(time,2)))
        EV_cnt_in_charge_queue -= 1
        EV_cnt_charging += 1
        
        # Update system performance measurement metrics
        
        data.serviced_EVs.append(client.id)
        data.EV_serviced += 1
        
        data.EV_waiting_delay.append({'time': time, 'wait_delay': time - client.arrival_time})
        data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
        
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        # Schedule completion of charging process
        FES.put((time + recharge_time, client.id, "Stand By"))
        
# Function to handle servicing of arriving customers
def customer_service(time, client_id, FES, cust_queue, standby_queue):
    global cust_cnt
    global EV_cnt_in_standby
    global data
    
    client_cust = cust_queue.pop(0)
    
    # If there are EVs on standby, the customer takes an EV and leaves. Else, it is missed service.
    if len(standby_queue) > 0:
        
        # FIFO based EV pick up
        client_EV = standby_queue.pop(0)
        #print(client_cust.id + ' takes ' + client_EV.id + ' and leaves at ' + str(time))
        
        # Update relevant tracking parameters
        cust_cnt -= 1
        EV_cnt_in_standby -= 1
        
        # Update system performance metrics
        
        data.serviced_custs.append(client_cust.id)
        data.Cust_serviced += 1
        
        data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
        data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
        
        data.Cust_missed_service_prob.append({'time': time, 'prob': round(data.Cust_unserviced / data.Custarr, 2)})
        
    else:
        # Missed service - update relevant tracking parameters
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
    
    electricity_prices = get_electricity_prices()
    
    # Define event queue    
    FES = PriorityQueue()
    
    # Initialize starting events
    time = 0
    FES.put((INTER_T, 'E0', "EV_arrival"))
    FES.put((CUST_T, 'C0', "Cust_arrival"))
    
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
                #print(client_id + ' misses service due to long wait time and leaves.')
                EV_cnt_in_charge_queue -= 1
                
                # Update system performance measurement metrics
                data.EV_reneged += 1
                data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
                
                data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                
                next
        
        # For other events, call the corresponding event handler function
        if event_type in ["EV_arrival", "Cust_arrival"]:
            arrival(client_id, time, FES, charge_queue, cust_queue)
        elif event_type == "Stand By":
            EV_charge_completion(time, client_id, FES, charge_queue, standby_queue)
        elif event_type in ['Cust_service']:
            customer_service(time, client_id, FES, cust_queue, standby_queue)

"""
Define support functions to help analyze and visualize the simulation results

"""

# Plots a performance metric over time
def plot_over_time(metric, title, ylabel):

    fig = plt.figure()
    ax = plt.axes()
    
    y = [t['avg_wait_delay'] for t in metric]
    x = [t['time'] for t in metric]
    
    ax.plot(x,y,'r')
    # plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.grid(True)
    
    plt.show()

"""
Define functions to identify warm-up transient period in performance metrics

"""

# Fishman's method - graphical method based on column means
def fishmans_method(avg_waiting_delay_list):
    
    metric = []
    time = []
    for i in range(len(avg_waiting_delay_list)):
        metric.append([x['avg_wait_delay'] for x in avg_waiting_delay_list[i]])
        time.append([x['time'] for x in avg_waiting_delay_list[i]])
    
    avg_metric = pd.DataFrame(metric).dropna(axis=1).mean().tolist()
    avg_time = pd.DataFrame(time).dropna(axis=1).mean().tolist()
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(avg_time, avg_metric,'r')
    # plt.title('Fishmans Method: Column Averages')
    plt.xlabel('Time')
    plt.ylabel('Average Wait Delay')
    plt.grid(True)
    
    plt.show()
    
    return avg_metric, avg_time

# Conway's rule - heuristic method based on comparison of each point to later parts of the simulation
def conways_rule(avg_waiting_delay):
    
    metric = [x['avg_wait_delay'] for x in avg_waiting_delay]
    time = [x['time'] for x in avg_waiting_delay]
    init_len = len(metric)
    while True:
        if (metric[0] != max(metric)) and (metric[0] != min(metric)):
            return time[init_len - len(metric)]
        else:
            metric.pop(0)

# Randomization test - statistical method based on significant differences bewteen sequential batches of values
def randomization_test(smoothed_metric, smoothed_time):
    
    N = int(len(smoothed_metric) / 20)
    avg_metric_batchmeans = [sum(smoothed_metric[x:min(len(smoothed_metric), (x+N))]) / len(smoothed_metric[x:min(len(smoothed_metric), (x+N))]) for x in range(0, len(smoothed_metric), N)]
    for i in range(1,len(avg_metric_batchmeans)):
        batch1 = avg_metric_batchmeans[:i]
        batch2 = avg_metric_batchmeans[i:]
        
        ttest, pval = ttest_ind(batch1, batch2)
        if pval > 0.05:
            break
        
    return smoothed_time[(i-1)*N]

"""
Functions to calculate confidence interval of a performance metric

"""
# Generic confidence interval function
def mean_CI(data, confidence=0.95):
    
    a = 1.0 * np.array(data)
    n = len(a) # 200
    m, se = np.mean(a), scipy.stats.sem(a)
    print("se", se)
    # h = standrad error * stats.t.ppf(confidence interval (95%))
    
    # we want h near 95 %
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print("h", h)
    return round(m,4), round(m-h,4), round(m+h,4),h

# Truncating transient period and calculating confidence interval of the mean of the performance metric
def confidence_interval(avg_waiting_delay_list, transient_cutoff_time = 30):
    
    metric = []
    time = []
    for i in range(len(avg_waiting_delay_list)):
        metric.append([x['avg_wait_delay'] for x in avg_waiting_delay_list[i]])
        time.append([x['time'] for x in avg_waiting_delay_list[i]])
    
    metric_means = []
    for i in range(len(metric)):        
        cutoff_idx = next(x for x, val in enumerate(time[i]) if val > transient_cutoff_time)
        metric_means.append(sum(metric[i][cutoff_idx:]) / len(metric[i][cutoff_idx:]))

    return mean_CI(metric_means)
    
"""
Single Simulation Run

"""

refresh_initial_params()
RANDOM_SEED = 42
data = Measure()
main_simulation(RANDOM_SEED)
avg_waiting_delay = data.EV_avg_waiting_delay
plot_over_time(metric = avg_waiting_delay, 
               title = 'Average waiting Time of EVs over time : Single Run', ylabel = 'Average Waiting Delay')

# Get point estimate of the performance metric
transient_cutoff_time = 30
cutoff_idx = next(x for x, val in enumerate([x['time'] for x in avg_waiting_delay]) if val > transient_cutoff_time)
point_estimate_mean = sum([x['avg_wait_delay'] for x in avg_waiting_delay][cutoff_idx:]) / len(avg_waiting_delay[cutoff_idx:])

"""
Multiple Independent Runs with different random seeds and different simulation times

"""

sim = [100, 200, 300, 500, 750, 1000]

print('\nConfidence Intervals of Average Wait Delay')
print('Format: (Expected Value, Lower CI, Upper CI)\n')

for SIM_TIME in sim:
       
    # Multiple Independent Runs
    avg_waiting_delay_list = []
    N_runs = 200
    for i in range(N_runs):
        RANDOM_SEED = int(random.random()*10000)
        
        INTER_T = 0.75
        CUST_T = 1        
        
        refresh_initial_params()
        data = Measure()
        main_simulation(RANDOM_SEED)
        avg_waiting_delay_list.append(data.EV_avg_waiting_delay)
    
    smoothed_metric, smoothed_time = fishmans_method(avg_waiting_delay_list)
    conways_rule(avg_waiting_delay)
    randomization_test(smoothed_metric, smoothed_time)
    
    print('Number of Simulation Runs = ' + str(N_runs) + ' and simulation time = ' + str(SIM_TIME))
    print(confidence_interval(avg_waiting_delay_list))
    print('')