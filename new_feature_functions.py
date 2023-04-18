import pandas as pd
import numpy as np
import schedule

def distance(loc1, loc2):
    return np.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

def trans_prob_matrix(ins): 
    # tpm[i,j] is the probability that, given a uniform random arrival moment of server in 
    #  time window of cust i, server arrives at cust j within its time window if the policy is to go to j
    n = ins.tasks.nr_tasks
    tw = ins.tasks.time_window
    s = ins.tasks.service_time
    tm = np.array(ins.tasks.travel_matrix)[1:, 1:]
    tpm = np.zeros((n,n)) # transition probability matrix
    for i in range(n):
        for j in range(n):
            if i==j: 
                continue
            
            start_i = tw[i][0] + s[i] + tm[i,j] # earliest arrival at customer j
            end_i = tw[i][1] + s[i] + tm[i,j] # latest arrival
            length = end_i - start_i
            start_j = tw[j][0]
            end_j = tw[j][1]
            if end_i <= start_j:
                tpm[i,j] = 0
            elif start_i <= start_j and start_j <= end_i and end_i <= end_j:
                tpm[i,j] = (end_i-start_j)/length
            elif start_i <= start_j and end_j <= end_i:
                tpm[i,j] = (end_j-start_j)/length
            elif start_j <= start_i and end_i <= end_j:
                tpm[i,j] = 1
            elif start_j <= start_i and start_i <= end_j and end_j <= end_i:
                tpm[i,j] = (end_j-start_i)/length
            elif end_j <= start_i:
                tpm[i,j] = 0
            else:
                raise ValueError("transition probability matrix malfunction")
    return tpm     

def near_neigh_matrix(distance):
    '''distance is a matrix, with distance[i,j] the distance between 
        i and j (np array). Returns matrix with element (i,j) being 
        the index of jth nearest neighbour of location i.
    '''
    return distance.argsort(axis=1)   
    
def tpm_metrics(tpm,ins,nn_matrix):
    N = tpm.shape[0]
    distances = np.array(ins.tasks.travel_matrix)[1:,1:]
    diag = distance(ins.tasks.region['pos'],
                    (ins.tasks.region['width'],ins.tasks.region['height']))
    d_tilde = (diag - distances)/diag
    D = np.zeros(N)
    D2 = np.zeros(N)
    extra_dist = np.ones(N)*diag  # How far do you have to travel from 
    #nearest neighbour to reach a feasible (time window wise) next customer?
    
    for i in range(N):
        D[i] = np.inner(d_tilde[i,:], tpm[i,:])
        for j in range(1,N):
            if tpm[i, nn_matrix[i, j] ] > 0.5 and j>1:
                extra_dist[i] = distances[i, nn_matrix[i, j]] - distances[i,nn_matrix[i,1]]
                break
        for k in range(1,4):
            D2[i] += tpm[i, nn_matrix[i,k] ]
            
    non_zero_in_D = np.count_nonzero(D)
    non_zero_in_D2 = np.count_nonzero(D2)
    sum_D = np.sum(D)
    sum_D2 = np.sum(D2)
    ext_dist = np.sum(extra_dist)
    
    return non_zero_in_D, non_zero_in_D2, sum_D, sum_D2, ext_dist

# using the seeds and corresponding groups from akkerman, can we form feasible 
# (time window wise) chains within each group? Count the number of customers
# that can't be included, in total.
def chain_metrics(clusters, distances, tpm, nn_matrix):
    N = tpm.shape[0]
    chains = [[cluster[0]] for cluster in clusters]
    for i, cluster in enumerate(clusters): 
        cluster2 = cluster.copy()
        seed = cluster[0]
        del cluster2[0]
        end = seed # index of end of chain
        len_chain = np.inf
        while len(chains[i]) != len_chain:
            len_chain = len(chains[i])
            for k in range(2,N):
                closest = distances[end, nn_matrix[end,1]]
                dist_kth_nn = distances[end, nn_matrix[end,k]] 
                kth_nearest = nn_matrix[end,k]
                if kth_nearest not in cluster2:
                    continue
                if tpm[end, kth_nearest] > 0.7:
                    chains[i].append(kth_nearest)
                    cluster2.remove(kth_nearest)
                    end = kth_nearest
                    
    nr_of_elements_in_chains = np.sum( [len(chain) for chain in chains] )
    return N - nr_of_elements_in_chains

def split_integer(M,N):
    '''divides M into N buckets of ints, all differing at most one,
      e.g. (34,5) ---> [7,7,7,7,6]'''
    buckets = [int(M/N)]*N
    shortage = M - sum(buckets) 
    buckets = [i+1 for i in buckets[:shortage]] + [i for i in buckets[shortage:]]
    return buckets
  
def workload_one_route(ins,route):
    '''calculates the estimated workload of the route of one shift'''
    S = ins.tasks.service_time
    N = len(route)
    dist = np.array(ins.tasks.travel_matrix)[1:,1:]
    load = 0
    for i in range(N-1):
        load += S[route[i]] + dist[route[i], route[i+1]]
    load += S[route[-1]]
    return load

def workload(ins,routes):
    loads = [workload_one_route(ins,route) for route in routes]
    return {'per route': loads, 'total':sum(loads)}

# for i in range(len(routes)):
#     workload(ins,routes[i])    

def load_based_split(ins):
    global rcount
    WL = ins.tasks.wl_est
    dist = np.array(ins.tasks.travel_matrix)[1:,1:]
    WL_part = WL/ins.shifts.nr_shifts
    S = ins.tasks.service_time
    windows = np.array(ins.tasks.time_window)
    order = windows[:, 1].argsort().tolist()
    routes = []
    route = [order[0]]
    accum_wl = S[order[0]]
    prev_loc = order[0]
    for loc in order[1:]:
        new_acc = accum_wl + dist[prev_loc, loc] + S[loc]
        if new_acc <= WL_part:
            route.append(loc)
            accum_wl = new_acc
            
        else:
            routes.append(route)
            route = [loc]
            accum_wl = S[loc]
        prev_loc = loc
    routes.append(route)
    
    if sum([len(route) for route in routes])!=len(S):
        raise ValueError("not all customers in a route")
    if len(routes)!=ins.shifts.nr_shifts:
        pass
        # rcount += 1
        # print("nr of routes: {}, nr of shifts: {}".format(len(routes),ins.shifts.nr_shifts))
    for route in routes:
        if workload_one_route(ins,route) > WL_part and len(route)>1:
            raise ValueError('load based split malfunction')
    return routes

def tw_of_routes(ins, routes):  #not important function
    tw = np.array(ins.tasks.time_window)
    timewindows = np.zeros((tw.shape[0], 2*len(routes)))
    for i,cluster in enumerate(routes):
        timewindows[:len(cluster),2*i:2*i+2] = tw[cluster]
    return timewindows

#twofroutes = tw_of_routes(ins,routes)

def wt_route_split(ins):
    windows = np.array(ins.tasks.time_window)
    N = ins.shifts.nr_shifts
    M = windows.shape[0]
    order = windows[:, 1].argsort().tolist()
    split = split_integer(M,N)
    routes = [[order.pop(0) for _ in range(split[i])] for i in range(N)]
    return routes


def FCFS_route(clusters,ins):
    windows = ins.tasks.time_window
    dist = np.array(ins.tasks.travel_matrix)[1:,1:] # delete [1:,1:]?
    routes = []
    for i,cluster in enumerate(clusters):
        if len(cluster)==1:
            continue
        
        windows_of_i = np.array(windows)[cluster,:]
        order_of_travel_indices = windows_of_i[:, 1].argsort()
        order_of_travel_cust = np.array(cluster)[order_of_travel_indices]
        routes.append(order_of_travel_cust.tolist())
    return routes

# this will create 0 waiting time1
def FCFS(ins):
    d = np.array(ins.tasks.travel_matrix)[1:,1:]
    tw = np.array(ins.tasks.time_window)
    order = tw[:, 1].argsort().tolist()
    tw = tw[order] # ordered time windows
    N = ins.shifts.nr_shifts
    S = np.array(ins.tasks.service_time)[order] # ordered service times
    routes = [[order[i]] for i in range(N)]
    serv_fin = [tw[i,0] + S[i] for i in range(N)] # next service finish times for each server
    next_server_to_finish = np.argmin(serv_fin)
    for i in range(N, tw.shape[0]):
        routes[next_server_to_finish].append(order[i])
        prev_loc = routes[next_server_to_finish][-2]
        serv_start = max(tw[i,0], serv_fin[next_server_to_finish] + d[prev_loc, order[i]]) # start time of next service
        serv_fin[next_server_to_finish] =  serv_start + S[i]
        next_server_to_finish = np.argmin(serv_fin)
        
    for i in range(d.shape[0]):
        if not np.any([i in r for r in routes]):
            raise ValueError('not in route', i)
        
    return routes


def eval_routes(routes, ins):
    route2 = [[i+1 for i in route] for route in routes]
    sched = schedule.Schedule(ins,route=route2)
    waiting_time = sched.waiting_time()
    total_travel = sched.distance()
    overtimes = sched.overtimes_per_shift()
    overtime = sched.shift_overtime()
    total = overtime+waiting_time+total_travel
    return {'waiting': waiting_time, 'overtime': overtime, 'travel': total_travel, 
            'total': total, 'shift overtimes':overtimes}

def modify_route(ins,routes):
    tw = np.array(ins.tasks.time_window)
    new_routes = []
    for route in routes:
        order = tw[route][:,1].argsort()
        new_route = np.array(route)[order].tolist()
        new_routes.append(new_route)
    return(new_routes)

# def time_window_spread(windows):
#     nr_grid_points = 100
#     grid = np.linspace(420,1380,nr_grid_points).tolist()
#     X = [0]*len(grid) # X[i] is the number of time windows which contain gridpoint i
#     for i,t in enumerate(grid):
#         for window in windows:
#             if t<=window[1] and t>=window[0]:
#                 X[i] += 1
                
#     X = np.array(X)/(nr_grid_points*len(windows))
#     return np.sum(X), np.std(X)

def time_window_spread(ins):
    tw = np.array(ins.tasks.time_window)
    midpoints = np.mean(tw, axis=1)
    midpoints.sort()
    grid = np.linspace(420,1320,len(midpoints)+2)[1:-1]
    return np.linalg.norm(grid-midpoints, ord=1)

def tw_gaps(ins):
    tw = np.array(ins.tasks.time_window)
    grid = np.linspace(420,1320,901)
    
# def time_window_dist_matrix(windows):
#     N = len(windows)
#     midpoints = [(window[1]+window[0])/2 for window in windows]
#     dist = np.zeros((N,N))
#     for i in range(N):
#         for j in range(i+1,N):
#             dist[i,j] = abs(midpoints[i]-midpoints[j])
#             dist[j,i] = dist[i,j]
#     return dist/np.max(dist)

def tw_spatial_spread(ins): #see photo
    windows = np.array(ins.tasks.time_window)
    N = windows.shape[0]
    midpoints = np.mean(windows, axis=1)
    S = ins.tasks.service_time
    dist = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            dist[i,j] = abs(midpoints[i]+S[i]-midpoints[j])
    delta = dist/np.max(dist) # time window 'distance' matrix
    distances = np.array(ins.tasks.travel_matrix)[1:,1:]
    d = distances/np.max(distances)
    min1 = np.zeros((N,N))
    D = np.zeros((N,N))
    c = np.sqrt(2)
    for i in range(N):
        for j in range(i+1,N):
            min1[i,j] = min(abs(d[i,j]-delta[i,j]),abs(d[j,i]-delta[j,i]))    
            D[i,j] = min(c, d[i,j]*min1[i,j])
    return np.sum(D)


# def tw_spatial_spread(tw_dist, distances):
#     dist_normalized = distances/np.max(distances)
#     return np.sum(abs(dist_normalized-tw_dist))

def tw_midpoints_std(ins, period):
    period_indx = get_tw_in_period(ins)[period]
    if period_indx == []:
        return 0
    tw = np.array(ins.tasks.time_window)[period_indx]
    midpoints = np.mean(tw, axis=1)
    return np.std(midpoints)



def get_tw_in_period(ins):
    tw = np.array(ins.tasks.time_window)
    mids = np.mean(tw,axis=1)
    early, mid, late = [],[],[]
    for i in range(tw.shape[0]):
        if mids[i] <= 720:
            early.append(i)
        elif mids[i] > 720 and mids[i] <= 1020:
            mid.append(i)
        else:
            late.append(i)
    return {'early': early, 'mid': mid, 'late': late, 'all':[i for i in range(tw.shape[0])]}
        
        

def MMs_wait(ins, period):
    ss = {'early': 420, 'mid': 720, 'late':1020}
    tw = np.array(ins.tasks.time_window)
    period_indx = get_tw_in_period(ins)[period]
    if period_indx == []:
        return 0
    tw = tw[period_indx]
    len_tw = tw[:,1]-tw[:,0]
    T = np.mean(len_tw)
    S = np.array(ins.tasks.service_time)[period_indx]
    A = ins.tasks.region['width']*ins.tasks.region['height']
    N = len(period_indx)
    ES = (sum(S) + 0.76*np.sqrt(N*A))/N
    s = ins.ss.count(ss[period]) # nr of servers
    capacity = ins.u[0]*s
    a = ES*N/capacity # is the 'load'
    rho = a/s
    pi_0 = ( np.sum( [a**j/np.math.factorial(j) for j in range(s)] ) + a**s/np.math.factorial(s)*1/(1-rho) )**-1
    alpha = 1/(1-rho) * a**s/np.math.factorial(s) * pi_0
    mu = 1/ES
    eta = mu*s*(1-rho)
    return alpha/eta*np.exp(-eta*T)*N
    
    
def GGs_wait(ins, period):
    ss = {'early': 420, 'mid': 720, 'late':1020}
    tw = np.array(ins.tasks.time_window)
    period_indx = get_tw_in_period(ins)[period]
    if len(period_indx)<3:
        return 0
    tw = tw[period_indx]
    len_tw = tw[:,1]-tw[:,0]
    T = np.mean(len_tw)
    S = np.array(ins.tasks.service_time)[period_indx]
    A = ins.tasks.region['width']*ins.tasks.region['height']
    N = len(period_indx)
    ES = (sum(S) + 0.76*np.sqrt(N*A))/N
    s = ins.ss.count(ss[period]) # nr of servers
    capacity = ins.u[0]*s
    a = ES*N/capacity # is the 'load
    rho = a/s
    pi_0 = ( np.sum( [a**j/np.math.factorial(j) for j in range(s)] ) + a**s/np.math.factorial(s)*1/(1-rho) )**-1
    alpha = 1/(1-rho) * a**s/np.math.factorial(s) * pi_0
    mu = 1/ES
    tw_sorted = tw[tw[:,0].argsort()]
    inter_arr_samples = [tw_sorted[i,0]-tw_sorted[i-1,0] for i in range(1,tw.shape[0])]
    ca2 = np.var(inter_arr_samples)/(np.mean(inter_arr_samples))**2
    cs2 = np.var(S)/(np.mean(S))**2
    eta = 2*s*(1-a)/(ES*(ca2+cs2))
    return alpha/eta*np.exp(-eta*T)*N
    

def est_nr_of_shifts(ins):
    estimated_wl = ins.tasks.wl_est
    estimated_nr_shifts = max(1,(estimated_wl/240))
    return {'estimated': estimated_nr_shifts, 'real': ins.shifts.nr_shifts}

def change_nr_shifts(ins, add):
    ins.shifts.shift_features['nr_shifts'] += add
    ins.shifts.shift_features['total_capacity'] += add*240
    ins.shifts.shift_output_data['nr_shifts'] += add
    ins.shifts.shift_output_data['start_time'].extend([0 for _ in range(add)])
    ins.shifts.shift_output_data['shift_length'].extend([240 for _ in range(add)])
    ins.shifts.nr_shifts += add
    ins.v += add
    ins.Q = np.hstack((ins.Q,np.ones((ins.n-1,1))))

def load_in_period(ins, period):
    if period == 'early':
        load = ins.shifts.estimates['early_load']
        shifts = ins.ss.count(420)
    elif period == 'mid':
        load = ins.shifts.estimates['mid_load']
        shifts = ins.ss.count(720)
    elif period == 'late':
        load = ins.shifts.estimates['late_load']
        shifts = ins.ss.count(1020)
    else:
        raise ValueError("load in period malfunction")
    if shifts == 0:
        shifts = 0.00001
    return load/(shifts*300)
