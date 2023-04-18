import numpy as np
import operator
from scipy.cluster.hierarchy import linkage
import schedule
import measures
import time
import datetime

"""
module contains algorithm for (generalized permutation) gomea

main function is gomea_solve: 
    executes gomea on instance (class object from Instance module)
    parameters are contained in specs dictionary
    set parameters manually with **params
    returns dictionary containing information on input and output
    
components:
    Individual class: methods and attributes for a schedule
        - key: encoding of schedule
        - score: evaluation of schedule
    
    Population class: pool of individuals (objects from Individual class)
        - buildTree: loads linkage-tree
        - nextGen: performs selection and variation procedure on population
            
note that linkage module builds linkage-tree based on minimum distance
while in description of gomea  linkage-tree is based on maximum distance
"""

#=================================SPECS========================================

'''
generations (integer): max number of generations
population (integer): population size
startpop (list or None type): option to manually input a start population of routes
threshold (float): threshold on ratio of score from last two generations
stop (integer): stops process if flat (counter) hits stop
'''

specs = {'generations': 20,
         'population': 200,
         'startpop': None,
         'threshold': 0.01,
         'stop': 2
        }

#==============================================================================


def gomea_solve(instance,**params):

    #initialization: loads input from **params if available else from specs dict. 
    getparam = lambda p : specs[p] if p not in params else params[p]
    pm = {}
    for key in specs:
        pm[key] = getparam(key)

    P = pm['population']
    G = pm['generations']
    startpop = pm['startpop']
    threshold = pm['threshold']
    stop = pm['stop']
    
    if startpop == None:
        models = [schedule.Schedule(instance) for i in range(P)]
        routes = [mod.route for mod in models]
        pop = Population(instance,routes)
    else:
        pop = Population(instance,params['startpop'])
    
    t = 0
    time_tracker = [0]
    g = 0
    prog = Progress(threshold,stop)
    prog.update(pop)
    while prog.go() and g < G:
        t0 = time.time()
        pop.generation = g
        pop.nextGen()
        prog.update(pop)
        t1 = time.time()
        print("evolution cycle %d finished in %s" % (g,str(datetime.timedelta(seconds=t1-t0))))
        t += t1 - t0
        time_tracker.append(t)
        g += 1

    bestIndIdx = np.argmin([ind.score for ind in pop.individuals])
    route = decode(pop.individuals[bestIndIdx].key,instance)
    mod = schedule.Schedule(instance,route)
   
    print('\n')
    print('total elapsed time:',str(datetime.timedelta(seconds=t)))
        
    result = {}
    result['params'] = pm #input parameters
    result['instance'] = instance.__dict__ #model parameters (class object)
    result['gen_count'] = g #number of generation cycles (int)
    result['time_track'] = time_tracker #time proceedings over generation cycles (list)
    result['route'] = mod.route 
    result['arrival'] = mod.arrival #arrival times (list of sublists)
    result['score'] = mod.evaluate() #performance (float)
    result['distance'] = mod.distance() 
    result['waiting_time'] = mod.waiting_time() 
    result['shift_overtime'] = mod.shift_overtime() 
    result['idle_time'] = mod.idle_time() 
    result['productivity'] = mod.productivity()
    result['progress'] = prog.progress #score progression over generations (list)
 
    return result

#keeps track of meta-parameters when to stop process
class Progress:
    def __init__(self,threshold,stop):
        self.progress = []
        self.flat = 0
        self.threshold = threshold
        self.stop = stop
    
    def ratio(self,x,y):
        if x == 0 and y == 0:
            return 0
        else:
            return abs(x-y)/max(x,y)
    
    def update(self,population):
        pop = population
        scores = [ind.score for ind in pop.individuals]
        topscore = min(scores)
        self.progress.append(topscore)
        if len(self.progress) > 2:
            if self.ratio(self.progress[-1],self.progress[-2]) <= self.threshold:
                self.flat += 1
            else:
                self.flat = 0
    
    def go(self):
        if self.flat < self.stop:
            return True
        else:
            return False
    
class Individual:
    def __init__(self,instance, route):
        self.instance = instance
        self.key, self.keyInt, self.keyDec = encode(route)
        self.score = evaluate(self.key,instance)

    def reencode(self):
        self.key, self.keyInt, self.keyDec = encode(decode(self.key,self.instance))

    def resize(self):
        if np.random.rand() > 0.1:
            self.reencode()
        
class Population:
    def __init__(self, instance, routes, tree=None):
        self.instance = instance
        self.individuals = [Individual(instance,route) for route in routes]
        self.size = len(self.individuals)
        self.tree = tree
        self.generation = 0

    def reencode(self):
        for individual in self.individuals:
            individual.reencode()

    def buildTree(self):
        self.tree = linkage(distances(self.individuals,self.instance), method='average')

    def nextGen(self):
        n = self.instance.n-1
        self.reencode()
        self.buildTree()        
        for individual in self.individuals:
            order = np.random.permutation(2 * n - 1)
            for i in order:
                candkey = individual.key + 0
                FOS = getset(self.tree, i, n)
                k = np.random.randint(0, self.size)
                for j in FOS:
                    candkey[j] = self.individuals[k].key[j]
                s = evaluate(candkey,self.instance)
                if s < individual.score:
                    individual.score = s
                    individual.key = candkey
                    individual.keyInt[FOS] = self.individuals[k].keyInt[FOS]
                    individual.keyDec[FOS] = self.individuals[k].keyDec[FOS]

#------------------------------------------------------------------------------
#support functions for Individual class
#------------------------------------------------------------------------------
         
def evaluate(key,instance):
    route = decode(key,instance)
    model = schedule.Schedule(instance,route=route)
    return model.evaluate()
    
def encode(route):
    route = schedule.adjust(route)
    n = sum([len(line) for line in route])
    key = np.zeros(n)
    keyInt = np.array(range(n))
    keyDec = np.zeros(n)
    for k in range(len(route)):
        r = np.random.uniform(size=len(route[k]))
        r.sort()
        key[route[k]] = r + k
        keyInt[route[k]] = k
        keyDec[route[k]] = r
    return key, keyInt, keyDec

def decode(key,instance):
    v = instance.v
    enum = list(enumerate(key))
    enum.sort(key=operator.itemgetter(1))
    route=[[] for k in range(v)]
    for r in range(len(key)):
        route[int(enum[r][1])].append(enum[r][0] + 1)
    return route

#------------------------------------------------------------------------------
#support functions for buildTree method
#------------------------------------------------------------------------------
    
def distance(i, j,individuals,instance):
    return 1-measures.full_depcy(i,j,individuals,instance,weight=2/3)
 
    
def distances(individuals,instance):
    d = []
    n = instance.n-1
    for i in range(n):
        for j in range(i + 1, n):
            d.append(distance(i, j,individuals,instance))
    return d

def getset(tree,i,n):
    if i < n:
        return [i]
    else:
        return getset(tree, int(tree[i - n, 0]),n) + getset(tree, int(tree[i - n, 1]),n)


if __name__ == "__main__":
    import instance
    ins = instance.get_ins()
    res = gomea_solve(ins,population=10)
    
    #import gantt
    #gantt.plt_gantt(res)
    #print("\007")
    