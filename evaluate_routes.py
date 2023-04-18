import numpy as np
import schedule

def eval_routes2(routes, ins): #this works
    windows = ins.tasks.time_window
    S = ins.tasks.service_time
    shift_length = ins.shifts.shift_length*2 ### remove *2
    dist = np.array(ins.tasks.travel_matrix)[1:,1:] # delete [1:,1:]?
    waiting_time = 0
    overtime = 0
    total_travel = 0
    overtimes_per_shift = []
    for i,cluster in enumerate(routes):
        if len(cluster)==1:
            continue
        ordered_S = np.array(S)[cluster]
        ordered_windows = np.array(windows)[cluster,:]
        time = ordered_windows[0,0] + ordered_S[0] + dist[cluster[0], cluster[1]] # does the index 0 occur in clusters dict?
        total_travel += dist[cluster[0], cluster[1]]
        time = max(time, ordered_windows[1,0])
        for j in range(1,ordered_windows.shape[0]-1):
            waiting_time += max(0, time - ordered_windows[j,1])
            time += ordered_S[j] + dist[cluster[j], cluster[j+1]]
            total_travel += dist[cluster[j], cluster[j+1]]
            time = max(time, ordered_windows[j+1,0])
        waiting_time += max(0, time - ordered_windows[-1,1])
        time += ordered_S[-1]
        overtimes_per_shift += [max(0, time - (ordered_windows[0,0]+shift_length[i]))] # if ins.shift.start_time not zero, problem
    
    overtime = sum(overtimes_per_shift)
    total = waiting_time + overtime + total_travel   
    return {'waiting': waiting_time, 'overtime': overtime, 'travel': total_travel, 
            'total': total, 'shift overtimes':overtimes_per_shift}
            
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
            
            
def eval_routes3(routes, ins): #this works
    starttimes = ins.ss
    windows = ins.tasks.time_window
    S = ins.tasks.service_time
    shift_length = ins.shifts.shift_length ### remove *2
    dist = np.array(ins.tasks.travel_matrix)[1:,1:] # delete [1:,1:]?
    waiting_time = 0
    overtime = 0
    total_travel = 0
    overtimes_per_shift = []
    for i,cluster in enumerate(routes):
        if len(cluster)==1:
            continue
        ordered_S = np.array(S)[cluster]
        ordered_windows = np.array(windows)[cluster,:]
        waiting_time += max(0, starttimes[i]-ordered_windows[0,1])
        time = max(starttimes[i],ordered_windows[0,0]) + ordered_S[0] + dist[cluster[0], cluster[1]] # does the index 0 occur in clusters dict?
        total_travel += dist[cluster[0], cluster[1]]
        time = max(time, ordered_windows[1,0])
        for j in range(1,ordered_windows.shape[0]-1):
            waiting_time += max(0, time - ordered_windows[j,1])
            time += ordered_S[j] + dist[cluster[j], cluster[j+1]]
            total_travel += dist[cluster[j], cluster[j+1]]
            time = max(time, ordered_windows[j+1,0])
        waiting_time += max(0, time - ordered_windows[-1,1])
        time += ordered_S[-1]
        overtimes_per_shift += [max(0, time - (starttimes[i]+shift_length[i]))] # if ins.shift.start_time not zero, problem
    
    overtime = sum(overtimes_per_shift)
    total = waiting_time + overtime + total_travel   
    return {'waiting': waiting_time, 'overtime': overtime, 'travel': total_travel, 
            'total': total, 'shift overtimes':overtimes_per_shift}

def FCFS_by_period(ins):
    d = np.array(ins.tasks.travel_matrix)[1:,1:]
    tw = np.array(ins.tasks.time_window)
    early, mid, late = [],[],[]
    nr_early, nr_mid, nr_late = ins.ss.count(420), ins.ss.count(720), ins.ss.count(1020)
    N = nr_early+nr_mid+nr_late
    
    order = tw[:, 1].argsort()
    tw = tw[order] # ordered time windows
    S = np.array(ins.tasks.service_time)[order]
    
    width = tw[10,1] - tw[10,0]
    for i in range(1,tw.shape[0]):
        overlap = max(tw[i-1,1] - tw[i,0], 0)
        if overlap/width > 0.9 and S[i] < .3*S[i-1]:
            tw[[i-1,i]] = tw[[i,i-1]]          
            order[[i-1,i]] = order[[i,i-1]] 
            S[[i-1,i]] = S[[i,i-1]] 
    
    
    route = [[order[i]] for i in range(nr_early)]
    serv_fin = [tw[i,0] + S[i] for i in range(nr_early)] # next service finish times for each server
    next_server_to_finish = np.argmin(serv_fin)    
    
    for i in range(nr_early, tw.shape[0]):       
        add_route_bool = nr_early+nr_mid<=len(route)<N and np.all( [j==np.inf for j in serv_fin[:nr_early+nr_mid]] )    
        if add_route_bool:
            route.append([order[i]])
            serv_fin.append(tw[i,0] + S[i])
        else:
            route[next_server_to_finish].append(order[i])
            prev_loc = route[next_server_to_finish][-2]
            serv_start = max(tw[i,0], serv_fin[next_server_to_finish] + d[prev_loc, order[i]]) # start time of next service
            serv_fin[next_server_to_finish] =  serv_start + S[i]

        if serv_fin[next_server_to_finish] > 720 and next_server_to_finish < nr_early and tw[i,1]>620:
            route[next_server_to_finish].pop()
            serv_fin[next_server_to_finish] = np.inf
            if len(route) < nr_early + nr_mid:
                route.append( [order[i]] )
                serv_fin.append(max(tw[i,0], 720) + S[i])
            else:
                first_mid_serv = np.argmin(serv_fin[nr_early:nr_early+nr_mid]) + nr_early  # first to finish among servers in afternoon
                route[first_mid_serv].append(order[i])
                prev_loc = route[first_mid_serv][-2]
                serv_start = max(tw[i,0], serv_fin[first_mid_serv] + d[prev_loc, order[i]])
                serv_fin[first_mid_serv] = serv_start + S[i]
        if serv_fin[next_server_to_finish] > 1020 and nr_early <= next_server_to_finish < nr_early + nr_mid and tw[i,1]>920:
            route[next_server_to_finish].pop()
            serv_fin[next_server_to_finish] = np.inf
            if len(route) < nr_early + nr_mid + nr_late:
                route.append( [order[i]] )
                serv_fin.append(max(tw[i,0], 1020) + S[i])
            else:
                first_late_serv = np.argmin(serv_fin[nr_early+nr_mid:nr_early+nr_mid+nr_late]) + nr_early+nr_mid
                route[first_late_serv].append(order[i])
                prev_loc = route[first_mid_serv][-2]
                serv_start = max(tw[i,0], serv_fin[first_mid_serv] + d[prev_loc, order[i]])
                serv_fin[first_mid_serv] = serv_start + S[i]

        next_server_to_finish = np.argmin(serv_fin)
        
    for i in range(d.shape[0]):
        if not np.any([i in r for r in route]):
            print('not in route', i)
            # raise ValueError('not in route', i)
            
    return route
            
