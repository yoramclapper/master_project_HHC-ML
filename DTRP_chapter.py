import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
import winsound
import random
import pickle
import json

#%% Functions
    
def draw_service_time(ES):
    #return np.random.normal(ES, math.sqrt(VarS))
    return np.random.exponential(ES)
    #return np.random.chisquare(ES)

def draw_coords():
    return width*np.random.uniform(), width*np.random.uniform()

def distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 )
    
    

def DES(lamb, ES, max_nr_of_arrivals):  # discrete event simulation
    # saving all arrivals in a table/list ###############
    arrival_moments = [0]   # moments of arrivals
    coords = [draw_coords()]  # coordinates of customers
    for t in range(1,max_nr_of_arrivals):
        Xt = np.random.exponential(1/lamb)
        x, y = draw_coords()
        coords.append((x,y))
        arrival_moments.append(arrival_moments[-1] + Xt)
    ####################################################
        
    time = 0
    S = draw_service_time(ES)
    next_service_finish_time = distance((.5, .5), coords[0]) + S
    queue_length = 0
    
    service_start_moments = [0]   # moments when serviceman starts driving to next cust
    arr_indx = 1  # n-th customer arrived in system so far
    cust_served = 0  # the index of the customer being served 
    waiting_times = []
    
    
    while cust_served < max_nr_of_arrivals:
        if arr_indx >= max_nr_of_arrivals: # block the queue
            next_arrival_moment = np.inf
        else:
            next_arrival_moment = arrival_moments[arr_indx]
    
        if next_service_finish_time < next_arrival_moment:
            time = next_service_finish_time
            position = coords[cust_served] # current position
            waiting_time = service_start_moments[cust_served] - arrival_moments[cust_served]
            waiting_times.append(waiting_time)
            if queue_length > 0:
                queue_length -= 1
            else:
                time = next_arrival_moment
                arr_indx += 1
            cust_served += 1
            if cust_served < max_nr_of_arrivals: 
                next_service_finish_time = time + draw_service_time(ES) + distance(position, coords[cust_served])
            service_start_moments.append(time)
        else:
            queue_length += 1
            time = next_arrival_moment
            arr_indx += 1
        
    return waiting_times

#%%

#### Main parameters ##################################################
max_nr_of_arrivals = 10**6
warmup = 10**4
batchsize = 1000
width = 0  # width of the square/region
ES = 1   # expected service time
VarS = 1   # variance service time
expected_distance = width*1/15*(2+math.sqrt(2)+5*math.log(1+math.sqrt(2))) # = 7*0.5214
var_distance = width**2*1/3 - expected_distance**2
total_var = VarS + var_distance
total_ES = ES + expected_distance
total_ES2 = total_ES**2 + total_var  # second moment
#######################################################################


excess_list = []
error_list = []
rel_excess_list = [] # relative excess (divide excess by PK)
rel_error_list = []
load_list =  [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

data = {'info': 'simulating each load 1e6 times, which is one batch, this 1000 times, keys are loads',
        'max_nr_of_arrivals': max_nr_of_arrivals, 'width': width}


for load in load_list:
    # waits_exp = []
    # for exp in range(1,51):
        # experiment = jj*50 + exp
        lamb = load/total_ES  # arival rate
        PK_predict = lamb*total_ES2/(2*(1-load))
        waiting_times = DES(lamb, ES, max_nr_of_arrivals)
        # waits_exp.append(waiting_times)
        # if exp%50==0:
        #     data['waits'] = waits_exp
        #     with open('data load {} exps {}.json'.format(load,experiment), 'w') as fp:
        #         json.dump(data, fp)
        #     print('experiment {}, load {}, jj {}'.format(experiment,load,jj))
        #     waits_exp = []
        
        ### Compute confidence interval #########################################
    
        simulation_mean = np.mean(waiting_times[warmup:])
        excess = simulation_mean - PK_predict
        nr_of_batch_means = int((max_nr_of_arrivals - warmup)/batchsize)
        batch_means = []
        waiting_times2 = waiting_times[warmup:]
        waiting_times3 = waiting_times2.copy()
        random.shuffle(waiting_times3)  # when using this, plotting mean so far won't work
        for i in range(nr_of_batch_means):
            batch = waiting_times3[i*batchsize:(i+1)*batchsize]
            batch_means += [np.mean(batch)]
            
        t_quantile = scipy.stats.t.ppf(0.975, nr_of_batch_means-1)
        half_width_CI = np.std(batch_means)/np.sqrt(nr_of_batch_means)*t_quantile
        CI = [np.mean(batch_means) - half_width_CI, np.mean(batch_means) + half_width_CI]
        ####################################################################
        
        x_axis = np.arange(min(batch_means),max(batch_means),0.01)
        plt.hist(batch_means, bins=max(int((nr_of_batch_means)/50),10), density=True)
        plt.axvline(PK_predict, color='r')
        plt.plot(x_axis, scipy.stats.norm.pdf(x_axis, PK_predict, np.std(batch_means)))
        plt.title('load={}, {} batches, std={:.3}'.format(load,nr_of_batch_means,np.std(batch_means)))
        plt.show()
        print(CI)
        
        excess_list.append(excess)
        error_list.append(half_width_CI)
        rel_error_list.append(half_width_CI/PK_predict)
        rel_excess_list.append(excess/PK_predict)
        
        
        mean = waiting_times2[0]
        mean_waiting_list = []
        for i in range(1,len(waiting_times2)):
            mean = mean + (waiting_times2[i] - mean)/(i+1)  # recursively computing the mean so far
            mean_waiting_list.append(mean)
    
        plt.plot(mean_waiting_list[int(.01*len(mean_waiting_list)):])
        plt.axhline(y=PK_predict, color='r')
        plt.xlabel("Customers")
        plt.ylabel("Running mean waiting time")
        plt.title("load = " + str(load))
        plt.savefig("load " + str(load) +".png", dpi=300, bbox_inches = "tight")
        plt.show()
        
        

    


plotit = True
if plotit:
    plt.errorbar(load_list, excess_list, yerr=error_list, capsize=5, fmt='bo-')
    plt.axhline(y=0, color='r')
    plt.xlabel("Load")
    plt.ylabel("Excess")
    plt.title("PK underestimation")
    # plt.savefig("PK1.png", dpi=300, bbox_inches = "tight")
    plt.show()
    plt.errorbar(load_list, rel_excess_list, yerr=rel_error_list, capsize=5, fmt='bo-')
    plt.axhline(y=0, color='r')
    plt.xlabel("Load")
    plt.ylabel("Relative Excess")
    plt.title("PK underestimation, relative")
    # plt.savefig("PK2.png", dpi=300, bbox_inches = "tight")
    plt.show()




        
        



