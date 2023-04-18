import numpy as np

"""
module contains Schedule class

input schedule class:
    instance: class object from Instance module (contains model parameters)
    route (list of sublists): arrival order at clients (excluding base) per shift
        - index of sublist is shift-id
        - sublists contain (possibly empty) subset of client-id's (in {1,2,...,n})
    arrival (list of sublists): arrival time at clients per shift
        - index of sublist is shift-id
        - sublist[1:-1] contains arrival time at clients (in order of schedule)
        - sublist[0] and sublist[-1] is start- and return time at base
    
    if set to None then attributes are generated automatically
    
contains methods to evaluate schedule:
    dist: total distance/time traveled
    tw_tard: total time window waiting time
    shift_tard: total shift overtime
    evaluate: score of schedule (weighted sum of above components)
"""

class Schedule:
    def __init__(self,instance,route=None, arrival=None):
        self.n = instance.n
        self.d = instance.d
        self.p = instance.p
        self.tw = instance.tw
        self.v = instance.v
        self.ss = instance.ss
        self.u = instance.u
        self.Q = instance.Q
        
        self.route = self.load_route(route)
        self.arrival = self.load_arrival(self.route,arrival)
        
    def load_route(self,route):
        if route == None:
            return random_route(self.n,self.v,self.Q)
        else:
            return route
        
    def load_arrival(self,route,arrival):
        if arrival == None:
            return get_arrival(self.route,self.d,self.p,self.tw,self.ss)
        else:
            return arrival
        
    def distance(self):
        total = 0
        for r in self.route:
            r = list(r)
            if len(r) > 0:
                r.insert(0,0)
                r.append(0)
                total += sum([self.d[r[i]][r[i+1]] for i in range(len(r)-1)])
            else:
                total += 0
        return total
        
    def waiting_time_per_shift(self):
        waits = []
        for r in self.route:
            if len(r) > 0:
                a = self.arrival[self.route.index(r)] 
                waits += [ sum([max(0,a[i+1]-self.tw[r[i]][1]) for i in range(len(r))]) ]
            else:
                waits += [0]
        return waits

    def waiting_time(self):
        return(sum(self.waiting_time_per_shift()))
    
    def overtimes_per_shift(self):
        overtimes = []
        for i,r in enumerate(self.route):
            if len(r) > 0:
               a = self.arrival[i] 
               duration = self.u[i]
               overtimes += [max(0,a[-1] - (self.ss[i] + duration))] #note: use self.ss[i] instead of a[0] when incl. shift start times
            else:
               overtimes += [0]
        return overtimes
                
        
    def shift_overtime(self):
        return(sum(self.overtimes_per_shift()))
        
    def idle_time(self):
        total = 0
        for r in self.route:
            if len(r) > 0:
                a = self.arrival[self.route.index(r)][1:-1]
                for i in range(len(r)-1):
                    j = i + 1
                    prev_arrival = a[i]
                    service_time = self.p[r[i]]
                    travel_time = self.d[r[i]][r[j]]
                    finish_time = prev_arrival + service_time + travel_time
                    next_arrival = a[j]
                    idle = max(0,next_arrival-finish_time)
                    total += idle
            else:
                total += 0
        return total
    
    def productivity(self):
        total_planned = sum(self.u)
        total_worked = self.distance() + sum(self.p)
        return total_worked/total_planned
    
    def evaluate(self):
        components = [2*self.distance(),
                      7*self.shift_overtime(),
                      self.waiting_time()]
        return sum(components) + 0*max(components) 
    
#------------------------------------------------------------------------------
#support functions to generate random feasible schedule
#------------------------------------------------------------------------------

#returns subset of clients that are feasible for shift k according to Q 
def get_feasible_clients(k,clients,Q):
    feasible = []
    for i in clients:
        if Q[i][k] == 1:
            feasible.append(i)
    return feasible

def random_route(n,v,Q):
    clients = list(range(n-1))
    shifts = list(range(v))
    adjshifts = list(range(v))
    route = [[] for k in shifts]
    
    while len(clients) > 0:
        getclients = [get_feasible_clients(k,clients,Q) for k in shifts]
        randomshift = np.random.choice(adjshifts)
        if len(getclients[randomshift]) == 0:
            adjshifts.remove(randomshift)
            continue
        randomclient = np.random.choice(getclients[randomshift])
        route[randomshift].append(randomclient+1)
        clients.remove(randomclient)

    return route

#------------------------------------------------------------------------------
#support functions to generate planning for schedule
#------------------------------------------------------------------------------

#returns arrival times for single route r
def r_arrival(r,d,p,tw,ss0):
    a = [max(ss0,tw[r[0]][0]-d[0][r[0]])]
    r = list(r)
    r.insert(0,0)
    r.append(0)
    for i in range(len(r)-2):
        a.append(max(a[i] + d[r[i]][r[i+1]] + p[r[i]], tw[r[i+1]][0]))
    a.append(a[-1] + d[r[-2]][r[-1]] + p[r[-2]])
    
    return a

def get_arrival(route,d,p,tw,ss):
    arrival = []
    for r in route:
        if len(r) > 0:
            ss0 = ss[route.index(r)]
            arrival.append(r_arrival(r,d,p,tw,ss0))
        else:
            arrival.append(None)
    return arrival

#------------------------------------------------------------------------------
#miscellaneous funtions
#------------------------------------------------------------------------------

#required in gomea module    
def adjust(schedule):
    sched = []
    for route in schedule:
        sched.append([i-1 for i in route])
    return sched

