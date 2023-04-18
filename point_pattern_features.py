import numpy as np


def distance(loc1, loc2):
    return np.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

def nearest_neighbour(cust, others):
    nearest_nb = (np.inf, np.inf)
    nearest_dist = np.inf
    for loc in others:
        if distance(cust, loc) < nearest_dist:
            nearest_dist = distance(cust, loc)
            nearest_nb = loc
            
    return nearest_nb, nearest_dist

def nearest_dist_list(locations):
    dist_list = []
    for i in range(len(locations)):
        loc = locations[i]
        other_locs = locations.copy()
        del other_locs[i]
        dist_list += [nearest_neighbour(loc, other_locs)[1]]
        
    return dist_list

def omega(ins, samples=60):
    locations = ins.tasks.location
    distances = []
    for _ in range(samples):
        random_point = np.random.uniform(0,ins.tasks.region["width"]), np.random.uniform(0,ins.tasks.region["height"])
        distances += [nearest_neighbour(random_point,locations)[1]]
    return np.mean(distances)

def mean_center(locations):
    x_coords = [point[0] for point in locations]
    y_coords = [point[1] for point in locations]
    return np.mean(x_coords), np.mean(y_coords)

def standard_dist(locations):
    mu_x, mu_y = mean_center(locations)
    x_coords = np.array( [point[0] for point in locations] )
    y_coords = np.array( [point[1] for point in locations] )
    dist1 = np.sum( (x_coords-mu_x)**2 + (y_coords-mu_y)**2 )
    return np.sqrt(dist1/len(locations))