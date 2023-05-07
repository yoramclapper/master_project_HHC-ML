'''create instances with est shift capacity seperately in morning, afternoon and evening.
also changed est_tt. Fixed tw length for all customers in one instance
'''

import numpy as np

"""
Module contains following classes to create instance object:
    - TaskParams : object contains specification of task parameters and methods to generate task data
    - ShiftParams: object contains specification of shift parameters
    - Instance : converts task, shift parameters and creates instance object
"""

class TaskParams:
    def __init__(self,nr_tasks):
        self.nr_tasks = nr_tasks
        
        #location parameters
        self.max_tt = np.random.uniform(5,15) #maximum travel time
        self.region = {"pos" : (0,0), "width" : self.max_tt*np.cos(np.pi/4), "height" : self.max_tt*np.sin(np.pi/4)} #defines rectangle
        self.grid = self.get_grid() #lays grid on rectangle
        self.grid_density = self.get_grid_density() #assigns density to grid
        
        #time window parameters
        self.length_tw = [30,60,120]
        self.length_dist = [0.5,0.3,0.2] #probability that tw has length [30,60,120]
        self.mins_tw = {"early" : 420,"mid" : 720,"late" : 1020} #minimal start time of tw in category (420 -> 7am)
        self.maxs_tw = {"early" : 720, "mid" : 1020,"late" : 1320} #maximal due time of tw in category (1380 -> 11pm)
        self.dist_tw = [0.5,0.05,0.45] #probability if tasks is [early,mid,late]
        
        #service time parameters for lognormal dist
        self.mean_st = 20.91
        self.std_st = 12.34
        
        #task attributes
        self.location = self.get_pos()
        self.service_time = self.get_service_time()
        self.time_window = self.get_tw()
        self.travel_matrix = self.get_travel_matrix()
        
        self.task_output_data = {
            "nr_tasks" : self.nr_tasks,
            "travel_matrix" : self.travel_matrix,
            "service_time" : self.service_time,
            "time_window" : self.time_window}
        
        #estimated workload for generated task data
        self.wl_est = self.estimate_workload() 
        
        #statistical features for generated task data
        self.task_features = self.get_task_features()
        
     
    #lays grid of (size x size) on region
    #returns rectangles on grid
    def get_grid(self,size=3):
        region = self.region
        x0, y0 = region["pos"]
        w = region["width"]
        h = region["height"]
        
        xstep = w/size
        ystep = h/size
        
        grid = []
        for n in range(size):
            for m in range(size):
                x = x0 + n*xstep
                y = y0 + m*ystep
                w = xstep
                h = ystep
                rec = {"pos" : (x,y), "width" : w, "height" : h}
                grid += [rec]
        
        return grid
    
    def get_grid_density(self):
        grid_weights = np.random.choice(range(101),size=9)
        total_weight = sum(grid_weights)
        return [weight/total_weight for weight in grid_weights]
    
    '''
    generates task by picking random rectangle on grid w.r.t. density
    and drawing random point on grid
    returns x, y coordinate of task
    '''
    def generate_pos(self):
        grid = self.grid
        density = self.grid_density
        rec = np.random.choice(grid, p=density)
        x0, y0 = rec["pos"]
        w = rec["width"]
        h = rec["height"]
        x = np.random.uniform(x0,x0+w)
        y = np.random.uniform(y0,y0+h)
        
        return x, y
    
    #generates coordinates of n tasks over whole region
    def get_pos(self):
        n = self.nr_tasks
        pos = []
        for _ in range(n):
            x, y  = self.generate_pos()
            pos += [(x,y)]
        
        return pos
    
    def euclid_dist(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    #returns travel matrix based on task coordinates in pos variable
    def get_travel_matrix(self):
        n = self.nr_tasks
        pos = self.location
        travel_matrix = []
        for i in range(n):
            travel_matrix += [[]]
            for j in range(n):
                p1 = pos[i]
                p2 = pos[j]
                dist = self.euclid_dist(p1,p2)
                travel_matrix[i] += [dist]
        for idx in range(n):
            travel_matrix[idx] = [0] + travel_matrix[idx]
        travel_matrix = [[0 for _ in range(n+1)]] + travel_matrix
        return travel_matrix
    
    #generates list of n random service times with given mean and std
    def get_service_time(self):
        n = self.nr_tasks
        mean = self.mean_st
        std = self.std_st
        mu = np.log(mean**2/np.sqrt(mean**2+std**2))
        sigma2 = np.log(1+(std**2/mean**2))
        service_time = []
        for _ in range(n):
            r = np.random.lognormal(mean=mu,sigma=np.sqrt(sigma2))
            service_time += [int(np.round(r,0))]
        return service_time
    
    '''
    generates list of n random time windows as follows (for each time window):
        (1) select random day part
        (2) select uniformly random midpoint of tw in interval corresponding to daypart 
        (3) select random length of time window
        (4) set feasible boundaries around midpoint of time window
    '''
    def get_tw(self):
        n = self.nr_tasks
        length_tw = self.length_tw
        length_dist = self.length_dist
        length = np.random.choice(length_tw,p=length_dist)
        minis = self.mins_tw
        maxis = self.maxs_tw
        dist = self.dist_tw
        keys = ["early", "mid", "late"]
        tw = []
        for _ in range(n): 
            random_key = np.random.choice(keys,p=dist)
            mini = minis[random_key]
            maxi = maxis[random_key]
            mid = np.random.uniform(mini,maxi)
            
            tw_start = max(420,mid-length/2) #420 corresponds with earliest start time 07.00
            tw_due = min(1320,mid+length/2) #1380 corresponds with latest end time 23.00
            tw += [(int(np.round(tw_start,0)),int(np.round(tw_due,0)))]
        return tw
    
    #rough estimate of total workload    
    def estimate_workload(self):
        total_st = np.sum(self.service_time)
        d = np.array(self.travel_matrix)[1:,1:]
        p = [i for i in range(d.shape[0])]
        est_tt = np.mean( [d[i,j] for i in p for j in p[p.index(i)+1:]] )
        est_work = (self.nr_tasks-1)*est_tt + total_st
        return int(round(est_work,0))
    
   
    #returns relative time window size (time window size / service time) statistics
    def get_tw_relsize_stat(self):
        relsizes = []
        for i in range(self.nr_tasks):
            tw = self.time_window[i]
            st = self.service_time[i]
            relsize = (tw[1] - tw[0])/st
            relsizes += [relsize]
        return (np.mean(relsizes),np.std(relsizes))
    
    #returns for each daypart category list of tws that fall in respective category
    def get_tw_cat(self):
        tw_early = []
        tw_mid = []
        tw_late = []
        for tw in self.time_window:
            midpoint = (tw[0] + tw[1])/2
            if midpoint >= self.mins_tw["early"] and midpoint < self.maxs_tw["early"]:
                tw_early += [tw]
            elif midpoint >= self.mins_tw["mid"] and midpoint < self.maxs_tw["mid"]:
                tw_mid += [tw]
            else:
                tw_late += [tw]
        return {'early' : tw_early, 'mid' : tw_mid, 'late' : tw_late}
                       
    #returns list of statistical features of generated task data
    def get_task_features(self):
        tw_cat = self.get_tw_cat()
        tw_early = tw_cat['early']
        tw_mid = tw_cat['mid']
        tw_late = tw_cat['late']
        d = np.array(self.travel_matrix)[1:,1:]
        p = [i for i in range(d.shape[0])]
        features = {
            'nr_tasks' : self.nr_tasks,
            'service_time_avg' : np.mean(self.service_time),
            'service_time_std' : np.std(self.service_time),
            'tw_relsize_avg' : self.get_tw_relsize_stat()[0],
            'tw_relsize_std' : self.get_tw_relsize_stat()[1], 
            'tw_early_freq' : len(tw_early),
            'tw_mid_freq' : len(tw_mid),
            'tw_late_freq' : len(tw_late),
            'distance_avg' : np.mean( [d[i,j] for i in p for j in p[p.index(i)+1:]] ),
            'distance_std' : np.std( [d[i,j] for i in p for j in p[p.index(i)+1:]] ),
            'estimated_workload' : self.wl_est
            }
        
        return features

#input 'shift_length' is list of shift durations       
class ShiftParams:
    def __init__(self,shift_length,start_time, estimates):
        
        #shift attributes
        self.nr_shifts = len(shift_length)
        self.start_time = start_time #set to 0 by default
        self.shift_length = shift_length
        self.estimates = estimates # estimated workload and nr of shifts for early, mid, late
        
        self.shift_output_data = {"nr_shifts" : self.nr_shifts,
                                  "start_time" : self.start_time,
                                  "shift_length" : self.shift_length}
        
        self.shift_features = {
            "nr_shifts" : self.nr_shifts,
            "total_capacity" : np.sum(self.shift_length)}
        
class Instance:
    def __init__(self,tasks,shifts):
        self.tasks = tasks
        self.shifts = shifts
        self.task_params = tasks.task_output_data
        self.shift_params = shifts.shift_output_data
        self.n = self.task_params["nr_tasks"] + 1 #+1 to add base location
        self.d = self.task_params["travel_matrix"]
        self.p = [0] + self.task_params["service_time"] #include base location
        self.tw = [None] + self.task_params["time_window"] #include base location
        self.v = self.shift_params["nr_shifts"]
        self.ss = self.shift_params["start_time"]
        self.u = self.shift_params["shift_length"]
        self.Q = self.load_Q() #qualification matrix
        self.feasibleShiftsForClients = self.det_feas_shifts_for_clients()

    def load_Q(self):   
        Q = np.zeros((self.n-1,self.v))
        for i in range(self.n-1):
            for k in range(self.v):
                    Q[i][k] = 1
        return Q
    
    def det_feas_shifts_for_clients(self):
        feasShiftsForClients = []
        for i in range(self.n - 1):
            feasShiftsForClients.append(np.argwhere(self.Q[i, :] == 1)[:, 0])
        return feasShiftsForClients

#creates random instance
def get_ins(min_task=30,max_task=60):
    nr_tasks = np.random.randint(min_task,max_task+1)
    tasks = TaskParams(nr_tasks)
    
    S = np.array(tasks.service_time)
    d = np.array(tasks.travel_matrix)[1:,1:]
    tw = np.array(tasks.time_window)
    
    # YC: split tasks by day part to define shifts by day part
    early_tasks, mid_tasks, late_tasks = [],[],[] 
    for i in range(nr_tasks):
        if tw[i,1] <= 720: 
            early_tasks.append(i)
        elif tw[i,0] < 720 and tw[i,1] > 720:
            if tw[i,1] - 720 < 720 - tw[i,0]:
                early_tasks.append(i)
            else: 
                mid_tasks.append(i)
        elif tw[i,0] >= 720 and tw[i,1] <= 1020:
            mid_tasks.append(i)
        elif tw[i,0] < 1020 and tw[i,1] > 1020:
            if tw[i,1] - 1020 < 1020 - tw[i,0]:
                mid_tasks.append(i)
            else:
                late_tasks.append(i)
        else:
            late_tasks.append(i)
            
    for i in range(nr_tasks): 
        if not np.any([i in early_tasks, i in mid_tasks, i in late_tasks]): 
            raise ValueError("early mid late split malfunction")
            
    early_mean_travel = np.mean( [d[i,j] for i in early_tasks for j in early_tasks[early_tasks.index(i)+1:]] )
    mid_mean_travel = np.nan_to_num ( np.mean( [d[i,j] for i in mid_tasks for j in mid_tasks[mid_tasks.index(i)+1:]] ) )
    late_mean_travel = np.mean( [d[i,j] for i in late_tasks for j in late_tasks[late_tasks.index(i)+1:]] )
    

    
    early_load = early_mean_travel*(len(early_tasks)-1) + np.sum(S[early_tasks])
    mid_load = mid_mean_travel*(len(mid_tasks)-1) + np.sum(S[mid_tasks])
    if np.isnan(mid_load):
        print('')
    late_load = late_mean_travel*(len(late_tasks)-1) + np.sum(S[late_tasks])
    
    est_nr_of_shifts_early = max(1,int(np.ceil(early_load/300)))
    est_nr_of_shifts_mid = max(1,int(np.ceil(mid_load/300)))
    est_nr_of_shifts_late = max(1,int(np.ceil(late_load/300)))
    
    estimates = {'est_nr_of_shifts_early': early_load/300,
                 'est_nr_of_shifts_mid': mid_load/300,
                 'est_nr_of_shifts_late': late_load/300,
                 'early_load': early_load,
                 'mid_load': mid_load,
                 'late_load': late_load,
                 'est_nr_of_shifts_total': tasks.wl_est/300}
    
    start_time = [420 for _ in range(est_nr_of_shifts_early)
                  ] + [720 for _ in range(est_nr_of_shifts_mid)
                       ] + [1020 for _ in range(est_nr_of_shifts_late)]
    shift_length = [300]*(est_nr_of_shifts_early + est_nr_of_shifts_mid + est_nr_of_shifts_late)
    shifts = ShiftParams(shift_length, start_time, estimates)
    
    return Instance(tasks,shifts)

if __name__ == "__main__":
    
    #load instance parameters
    instance = get_ins()
    
    n = instance.n
    d = instance.d
    p = instance.p
    tw = instance.tw
    v = instance.v
    ss = instance.ss
    u = instance.u
    Q = instance.Q
    
    summary = {
        "n" : n,
        "d" : np.array(d),
        "p" : p,
        "tw" : tw,
        "v" : v,
        "ss" : ss,
        "u " : u,
        "Q" : np.array(Q)}

    print('\n')
    for key, val in instance.tasks.task_features.items():
        print(key+" : ")
        print(val)
        print('\n')
            
    for key, val in instance.shifts.shift_features.items():
        print(key+" : ")
        print(val)
        print('\n')

    