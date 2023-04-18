#%% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sklmetrics
import pickle
import schedule
import gomea
import copy
import evaluate_routes
import plotly.express as px
from evaluate_routes import *

metrics = ['total_waiting_time', 'total_travel_time','total_overtime','total_idle_time']
# data_indices = [26,27,28]
# data_indices = [31]#[26,27,28,29,30]
data_indices = [35]#,36,37] # all customers have same tw length

#%%
for data_indx in data_indices:
    dataset_str = 'dataset_gom' + str(data_indx) + '.pickle'
    with open(dataset_str, 'rb') as handle:
        data = pickle.load(handle)
    if dataset_str == 'dataset_gom{}.pickle'.format(data_indices[0]):
        dataset = data.copy()
    else:
        dataset['instance'] += data['instance']
        for metric in metrics:
            dataset[metric] += data[metric]

del data
instances = dataset['instance']
dataset.pop('instance')
dataset.pop('info')


class Inst:
    def __init__(self, ins):
        self.tasks = ins['tasks']
        self.shifts = ins['shifts']
        self.n = ins['n']
        self.d = ins['d']
        self.p = ins['p']
        self.tw = ins['tw']
        self.v = ins['v']
        self.ss = ins['ss']
        self.u = ins['u']
        self.Q = ins['Q']
        self.feasibleShiftsForClients = ins['feasibleShiftsForClients']
        
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
    
def tw_of_routes(ins, routes): 
    scores = eval_routes(routes,ins)
    shift_overtimes = scores['shift overtimes']
    waiting_times = scores['waiting times']
    tw = np.array(ins.tasks.time_window)
    timewindows = np.zeros((tw.shape[0], 2*len(routes)))
    for i,cluster in enumerate(routes):
        timewindows[0,2*i] = ins.ss[i]
        timewindows[1,2*i] = shift_overtimes[i]
        timewindows[2,2*i] = waiting_times[i]
        timewindows[3:len(cluster)+3,2*i:2*i+2] = tw[cluster]
    print(scores)
    columns = np.repeat([i+1 for i in range(len(routes))],2).tolist()
    rows = ['shift start', 'overtime', 'waiting'] + ['tw {}'.format(i) for i in range(timewindows.shape[0]-3) ]
    df = pd.DataFrame(timewindows, rows, columns)
    #df[df.eq(0)] = ''
    return df

def service_time_of_routes(ins, routes):  #not important function
    
    S = np.array(ins.tasks.service_time)
    matrix = np.zeros((S.shape[0], len(routes)))
    for i,cluster in enumerate(routes):
        matrix[0,i] = np.sum(S[cluster])
        matrix[1:len(cluster)+1,i] = S[cluster]
    columns = [i+1 for i in range(len(routes))]
    return pd.DataFrame(matrix, columns=columns)

def eval_routes(routes, ins):
    route2 = [[i+1 for i in route] for route in routes]
    sched = schedule.Schedule(ins,route=route2)
    waiting_time = sched.waiting_time()
    total_travel = sched.distance()
    overtimes = sched.overtimes_per_shift()
    waiting_times = sched.waiting_time_per_shift()
    overtime = sched.shift_overtime()
    total = overtime+waiting_time+total_travel
    return {'waiting': waiting_time, 'overtime': overtime, 'travel': total_travel, 
            'total': total, 'shift overtimes':overtimes, 'waiting times': waiting_times}

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

def load_graph(ins):
    early, mid, late = ins.ss.count(420), ins.ss.count(720), ins.ss.count(1020)
    tw = np.array(ins.tasks.time_window)
    grid = np.linspace(420,1320,901).tolist()
    counts = []
    for i,x in enumerate(grid):
        count = 0
        for j in range(tw.shape[0]):
            if tw[j,0] <= x and x <= tw[j,1]:
                count += 1
        counts.append(count)
    
    plt.plot(grid,counts)
    plt.xticks([420+90*i for i in range(11)])
    plt.yticks(np.arange(0,max(counts),1))
    [plt.axvline(i, linestyle='--', color='r') for i in [420,720,1020]]
    plt.axhline(early,0.05,1/3, linestyle=':')
    plt.axhline(mid,1/3,2/3, linestyle=':')
    plt.axhline(late,2/3,.95, linestyle=':')
    plt.xlabel('time')
    plt.ylabel('workload')
    plt.title('Counting time window overlaps')
    plt.tight_layout()
    plt.savefig('tw based workload.png', dpi=300)
    plt.show()
    return np.array([grid,counts]).T

def load_graph2(ins):
    early, mid, late = ins.ss.count(420), ins.ss.count(720), ins.ss.count(1020)
    tw = np.array(ins.tasks.time_window)
    arrivals = np.mean(tw, axis=1)
    S = ins.tasks.service_time
    grid = np.linspace(420,1320,901).tolist()
    workload = []
    for i,t in enumerate(grid):
        count = 0
        for j in range(tw.shape[0]):
            if t >= arrivals[j] and t <= arrivals[j]+S[j]:
                count += 1
        workload.append(count)
    plt.plot(grid,workload)
    plt.xticks([420+90*i for i in range(11)])
    plt.yticks(np.arange(0,max(workload),1))
    [plt.axvline(i, linestyle='--', color='r') for i in [420,720,1020]]
    plt.axhline(early,0.05,0.05+1/3*.9, linestyle=':')
    plt.axhline(mid,1/3,2/3, linestyle=':')
    plt.axhline(late,2/3,.95, linestyle=':')
    plt.xlabel('time')
    plt.ylabel('workload')
    plt.title('Counting service-based interval overlaps')
    plt.tight_layout()
    plt.savefig('service based workload.png', dpi=300)
    plt.show()
    return np.array([grid,workload]).T



df = pd.DataFrame(dataset) 

#label data
features_metrics = list(df)
features = []
for attribute in features_metrics:
    if attribute not in metrics:
        features += [attribute]
df = df[features+metrics]

df2 = df[metrics]
df2.insert(len(list(df2)), 'total score', df['total_travel_time']+df['total_overtime']+df['total_waiting_time'])
df2.insert(0, 'ins index', [i for i in range(len(instances))])
df_big_waiting = df2.sort_values(by='total_waiting_time', ascending=False)
df_big_overtime = df2.sort_values(by='total_overtime', ascending=False)
df_big_total = df2.sort_values(by='total score', ascending=False)
diff_ins_sorted = df_big_overtime['ins index'].tolist()[:250]

#%%
# maximum_overtimes = []
# for i in df_big_overtime['ins index'].tolist():
#     diff_ins = copy.deepcopy(Inst(instances[i]['instance']))
#     sol = instances[i]
#     best_route = [[i-1 for i in route] for route in sol['route']]
#     scores = eval_routes(best_route,diff_ins)
#     maximum_overtimes.append(max(scores['shift overtimes']))

#%% time windows matrix and workload graph
insindex = df_big_waiting['ins index'].tolist()[101]
# insindex = 200
diff_ins = copy.deepcopy(Inst(instances[insindex]['instance']))  # difficult instance
sol = instances[insindex]
best_route = [[i-1 for i in route] for route in sol['route']]
tw_routes = tw_of_routes(diff_ins, best_route)
serv_routes = service_time_of_routes(diff_ins,best_route)
counts = load_graph(diff_ins) 
counts2 = load_graph2(diff_ins)
tw = np.array(diff_ins.tasks.time_window)
order = tw[:,0].argsort()
tw = tw[order]
ins = diff_ins




#%% plot routes

ins = diff_ins
waiting_time = sol['waiting_time']
overtime = sol['shift_overtime']
location = np.array(ins.tasks.location)
tw = np.array(ins.tasks.time_window)
color = ['b','k','r','m','c','g']
print(ins.ss)
for i,cluster in enumerate(best_route[0:]):
    col = color[i]
    loc = location[cluster]
    tw2 = tw[cluster]
    plt.quiver(loc[:-1,0], loc[:-1,1], loc[1:,0]-loc[:-1,0], loc[1:,1]-loc[:-1,1], scale_units='xy', angles='xy', scale=1, color=col, width=0.003)
    for j in range(len(loc)):
        plt.text(loc[j,0]+0.05,loc[j,1],str(tw2[j]), fontsize = 4, c=col)
# plt.title("waiting time {:.4},   overtime {:.4}".format(float(waiting_time),float(overtime)))
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.tight_layout()
plt.savefig('quiverplot routes.png', dpi=300)
plt.show()

#%%

def time(x):
    x = int(x)
    hours = int(x/60)
    minutes = x-hours*60
    if hours<24:
        return '1970-01-01 {:02d}:{:02d}:00'.format(hours,minutes)
    else :
        return '1970-01-02 {:02d}:{:02d}:00'.format(hours-24,minutes)


def gantt(ins,arr,route,name,wait):
    S = np.array(ins.tasks.service_time)
    tw = np.array(ins.tasks.time_window)
    tw_start = np.array( [time(tw[i,0]) for i in range(tw.shape[0])] )
    tw_end = np.array( [time(tw[i,1]) for i in range(tw.shape[0])] )
    df_data1, df_data2, df_data3 = [],[],[]
    for i,r in enumerate(route):
        tw_rs = tw_start[r]
        tw_re = tw_end[r].tolist()
        for j in range(len(r)):
            df_data1.append( dict(Task='{} Shift {}'.format(j+1,i+1), Start=str(tw_rs[j]), 
                                Finish=tw_re[j], Resource="Time window shift "+str(i+1), S=S[r][j], cust=r[j]) )
            df_data2.append( dict(Task='{} Shift {}'.format(j+1,i+1), Start=time(arr[i][j+1]), 
                                 Finish=time(arr[i][j+1]+3), Resource="Arrival", S='N/A' ) )
            df_data3.append( dict(Task='{} Shift {}'.format(j+1,i+1), Start=time(arr[i][j+1]+S[r][j]), 
                                 Finish=time(arr[i][j+1]+S[r][j]+3), Resource="Departure", S='N/A' ) )
            
    df_data = df_data1 + df_data2 + df_data3
    df = pd.DataFrame(df_data)
    df.sort_values(by='Resource')
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", 
                      hover_data=['S','cust'], title='waiting time {:.4}'.format(wait)
                     )
    fig.add_vline(x='1970-01-01 12:00:00', line_width=3, opacity=.5)
    fig.add_vline(x='1970-01-01 17:00:00', line_width=3, opacity=.5)
    
    fig.update_layout(xaxis=dict(
                          title='Timestamp', 
                          tickformat = '%H:%M',
                      ))
    fig.write_html("C:\\Users\\elham\\OneDrive - Vrije Universiteit Amsterdam\\MSc_Study\\Master Project\\forecasting-hhcrsp\\file {}.html".format(name), auto_open=True)
    fig.show()



#%%

            
# route = FCFS_by_period(ins)
# arr = schedule.Schedule(ins,[[i+1 for i in r] for r in route]).arrival
# print(eval_routes(best_route,ins),'\n')
# print(eval_routes(route,ins))
# wait_new = eval_routes(route,ins)['waiting']
# wait_best = eval_routes(best_route,ins)['waiting']

# gantt(ins,sol['arrival'], best_route, 'best',wait_best)
# gantt(ins,arr,route,'new',wait_new)

# d = np.array(ins.tasks.travel_matrix)[1:,1:]


with open('mytable.tex', 'w') as tf:
     tf.write(tw_routes.to_latex())










