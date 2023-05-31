import numpy as np
from scipy.stats import binom

'''
module contains dependency measures and methods that help build them
'''

#==============================================================================
#support functions
#==============================================================================

#returns probability that i and j are on the same shift in a random schedule
def same_shift_prob(i,j,instance):
    v = instance.v
    ishifts = instance.feasibleShiftsForClients[i]
    jshifts = instance.feasibleShiftsForClients[j]
    ni = len(ishifts)
    nj = len(jshifts)
    pi = np.zeros(v)
    pj = np.zeros(v)
    for k in ishifts:
        pi[k] = 1/ni
    for k in jshifts:
        pj[k] = 1/nj
    return np.dot(pi,pj)

#returns number of same shifts scheduling for indices i and j
def same_shift(i,j,individuals):
    size = len(individuals)
    x = 0
    for k in range(size):
        ri = individuals[k].keyInt[i]
        rj = individuals[k].keyInt[j]
        if ri == rj:
            x += 1
    return x

#returns joint/marginal distributions of shift distribution for i and j
def shift_distribution(i,j,individuals,instance):
    size = len(individuals)
    v = instance.v
    bins = np.zeros((v,v))
    for k in range(size):
        ri = individuals[k].keyInt[i]
        rj = individuals[k].keyInt[j]
        bins[(ri, rj)] += 1
    xyjoint = bins/size
    xmarginal = np.array([np.sum(xyjoint[k,:]) for k in range(v)])
    ymarginal = np.array([np.sum(xyjoint[:,l]) for l in range(v)])
    return xyjoint, xmarginal, ymarginal

#returns number of times j follows i if on same shift       
def relative_order(i,j,individuals):
    size = len(individuals)
    x = 0
    for k in range(size):
        si = individuals[k].keyInt[i]
        sj = individuals[k].keyInt[j]
        ri = individuals[k].key[i]
        rj = individuals[k].key[j]
        if si == sj and ri < rj:
            x += 1
    return x

#returns total sum of squared differences for random keys on index i and j if on same shift
def relative_adjacency(i,j,individuals):
    size = len(individuals)
    sqsum = 0
    for k in range(size):
        ri = individuals[k].keyInt[i]
        rj = individuals[k].keyInt[j]
        if ri == rj:
            sqsum += (individuals[k].key[i]-individuals[k].key[j])**2
    return sqsum

#==============================================================================
#statistical measures
#==============================================================================

binom_cdf_storage = {}  # a bit hacky, but this stores binom.cdf's
def binomial(x,size,p):
    E = size*p
    
    # retrieve or determine numerator
    if (x, size, p) in binom_cdf_storage:
        num = binom_cdf_storage[(x, size, p)]
    else:
        num = binom.cdf(x, size, p)
        binom_cdf_storage[(x, size, p)] = num  # store it for re-use

    # retrieve or determine denominator
    if (E, size, p) in binom_cdf_storage:
        denom = binom_cdf_storage[(E, size, p)]
    else:
        denom = binom.cdf(E, size, p)
        binom_cdf_storage[(E, size, p)] = denom  # store it for re-use

    if x <= E:
        return num/denom
    else:
        return (1 - num)/(1 - denom)

#==============================================================================
#entropy measures
#==============================================================================

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def mutual_info(xyjoint,xmarginal,ymarginal,normalizer):
    mi = 0
    normalizerTerm = np.log(normalizer)
    for k, px in enumerate(xmarginal):
        if px == 0:
            continue
        for l, py in enumerate(ymarginal):
            if py == 0:
                continue
            pxy = xyjoint[(k, l)]
            if pxy > 0 and pxy/(px*py) != 1:
                mi += pxy*np.log(pxy/(px*py))/normalizerTerm
    return mi

#==============================================================================
#dependency measures
#==============================================================================

def random_depcy():
    return 1

def inter_depcy(i,j,individuals,x):
    if x == 0:
        return 0
    else:
        p = relative_order(i,j,individuals)/x
        avg_sqdiff = relative_adjacency(i,j,individuals)/x
        return (1-entropy(p))*(1-avg_sqdiff)    

def full_depcy(i,j,individuals,instance,weight):
    size = len(individuals)
    x = same_shift(i,j,individuals)
    p = same_shift_prob(i,j,instance)
    E = size*p
    bi = 1-binomial(x,size,p)
    xyjoint, xmarginal, ymarginal = shift_distribution(i,j,individuals,instance)
    normalizer = min(len(instance.feasibleShiftsForClients[i]),
                     len(instance.feasibleShiftsForClients[j]))
    mi = mutual_info(xyjoint,xmarginal,ymarginal,normalizer)
    inter = inter_depcy(i,j,individuals,x)
    if x <= E:
        return bi*(weight+(1-weight)*mi)
    else:
        return bi*(weight+(1-weight)*inter)   