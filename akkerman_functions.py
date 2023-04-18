import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial import ConvexHull
import copy
from evaluate_routes import *
from new_feature_functions import *


def enclosing_rectangle(locations): # Depot assumed to be nowhere&everywhere!
    xvals = [location[0] for location in locations]
    yvals = [location[1] for location in locations]
    width = max(xvals) - min(xvals)
    height = max(yvals) - min(yvals)
    centroid = (min(xvals) + 0.5*width, min(yvals) + 0.5*height)
    lower_left = (min(xvals),min(yvals))
    return {'height': height, 'width': width, 'area': width*height, 
            'perimeter': 2*width+2*height, 'centroid': centroid, 'lowerleft': lower_left}

def inner_distances(locations):
    N = np.array(locations).shape[0]
    return [distance(locations[i], locations[j]) 
                     for i in range(N) for j in range(i+1,N)] 

def distances_to_point(locations, center):
    return [distance(p,center) for p in locations]

def cust_centroid(locations):
    return np.mean([p[0] for p in locations]), np.mean([p[1] for p in locations])

def variance_of_angles(locations, point):
    delta_xs = [p[0]-point[0] for p in locations]
    delta_ys = [p[1]-point[1] for p in locations]
    angles = np.arctan2(delta_ys, delta_xs)
    return np.var(angles)

# angles = variance_of_angles(locations,enclosing_rectangle(locations)['centroid'])

def nr_of_cust_within_circle(locations, radius_factor, center, hull):
    vertices = np.array(locations)[hull.vertices]
    M = max( [distance(vertex, center) for vertex in vertices] )

    if len([cust for cust in locations if distance(cust, center)<=M]) != len(locations):
        raise ValueError("Hull didn't work for f23-f28")
    return len([cust for cust in locations if distance(cust, center)<=radius_factor*M])
    
def subrectangles(locations, big_rectangle, partitions):
    def inside_subrectangle(cust, subrectangle):
        return (cust[0]>=subrectangle['lowerleft'][0] and cust[1]>=subrectangle['lowerleft'][1] 
               and cust[0]<=subrectangle['upperright'][0] and cust[1]<=subrectangle['upperright'][1])
    
    x0, y0 = big_rectangle['lowerleft']
    width = big_rectangle['width']
    height =  big_rectangle['height']
    subrectangle = {(i,j):{} for i in range(1,partitions+1) for j in range(1,partitions+1)}
    for i in range(1,partitions+1):
        for j in range(1,partitions+1):
            sublowerleft = (x0 + (i-1)*width/partitions, y0 + (j-1)*height/partitions)
            subupperright = (x0 + i*width/partitions, y0 + j*height/partitions)
            subrectangle[i,j]['lowerleft'] = sublowerleft
            subrectangle[i,j]['upperright'] = subupperright
            subrectangle[i,j]['centroid'] = ((sublowerleft[0]+subupperright[0])/2, 
                                             (sublowerleft[1]+subupperright[1])/2)
                                             
            for loc in locations:
                if inside_subrectangle(loc, subrectangle[i,j]):
                    subrectangle[i,j]['activated'] = True
                    break
            else:
                subrectangle[i,j]['activated'] = False
    
    mean_centroid_dist_activ = np.mean( [distance(subrectangle[rec1]['centroid'], 
                                subrectangle[rec2]['centroid']) for rec1 in subrectangle 
                                for rec2 in subrectangle if (rec1!=rec2 
                                and subrectangle[rec1]['activated'] and subrectangle[rec2]['activated'])] )
    dist_within_recs = []      
    for i, rec2 in subrectangle.items():
        rec = subrectangle[i]
        if rec['activated']:
            cust_in_rectangle = [loc for loc in locations if inside_subrectangle(loc,rec)]
            dist_within_recs.extend( inner_distances(cust_in_rectangle) )
    mean_dist_cust_within_recs = np.mean(dist_within_recs)
    if dist_within_recs == []:
        mean_dist_cust_within_recs = 0
                                                         
    return {"mean dist activ recs": mean_centroid_dist_activ, 
            "mean dist cust within recs": mean_dist_cust_within_recs, 'recs':subrectangle}

def seeds(locations, nr_shifts, hull):
    '''select nr_shifts points that are furthest apart (f44-f48)'''
    hull_vertices = np.array(locations)[hull.vertices]
    ## select seeds from the hull by going around it and skipping two vertices 
    seeds = [hull_vertices[i] for i in range(0,len(hull_vertices)-2,3)]
    seed_indices = [int(hull.vertices[i]) for i in range(0,len(hull_vertices)-2,3)]
    remaining_points = np.delete(np.array(locations), seed_indices, axis=0)
    remaining_points_indices = [i for i in range(len(locations)) if i not in seed_indices]
    shortage = nr_shifts - len(seeds)
    if shortage<0:
        del seeds[0:abs(shortage)]  
    for i in range(shortage):
        smallest_dist = 0
        for j in range(remaining_points.shape[0]):
            point = remaining_points[j]
            min_dist = min( [distance(point, seed) for seed in seeds] ) # distance to closest seed
            if min_dist > smallest_dist:
                new_seed_indx = remaining_points_indices[j]
                new_seed = point
                smallest_dist = min_dist
                indx = j
        seed_indices.append(new_seed_indx)
        seeds.append(new_seed)
        remaining_points = np.delete(remaining_points, indx, axis=0)
        del remaining_points_indices[indx]
        
    if len(seeds)!=nr_shifts:
        raise ValueError("seed function defect")

    return np.array(seeds), remaining_points, seed_indices, remaining_points_indices

def clustering(seeds, remaining_points, seed_indices, remaining_points_indices):
    clusters = {i: [seeds[i]] for i in range(seeds.shape[0])}
    clusters_indx = [[seed_indices[i]] for i in range(seeds.shape[0])]
    for i in range(remaining_points.shape[0]):
        point = remaining_points[i]
        point_indx = remaining_points_indices[i]
        indx_of_closest_seed = np.argmin( [distance(point, seed) for seed in seeds] )
        clusters[indx_of_closest_seed].append(point)
        clusters_indx[indx_of_closest_seed].append(point_indx)
        
    distances_within_cluster = []
    distances_to_centroid = []
    centroids = []
    for i in clusters:
        
        clusters[i] = np.array(clusters[i])
        members = clusters[i]
        distances_within_cluster.extend( inner_distances(members) )
        centroids.append(cust_centroid(members))
        distances_to_centroid.extend( distances_to_point(members, cust_centroid(members)) )
        
    mean_dist_between_centroids = np.mean( inner_distances(centroids) )
    if np.isnan(mean_dist_between_centroids):
        pass
        #print("clustering function error", clusters)    
                                                    
    return (clusters, clusters_indx, np.mean(distances_within_cluster), mean_dist_between_centroids, 
            np.mean(distances_to_centroid))

def get_akkerman_features(ins, only_clust=False):
    indices = [i for i in range(1,49) if i not in [9,10,11,14,19,23,24,29,30,33,34,37,38,39,40,42,43,47,48]]
    locations = ins.tasks.location
    hull = ConvexHull(np.array(locations))
    seed_list, remaining_points, seed_indices, remaining_points_indices = seeds(locations, ins.shifts.nr_shifts,hull)
    (clusters, clusters_indx, mean_distance_within_clusters, mean_dist_between_centroids, 
            mean_distance_to_centroid) = clustering(seed_list,remaining_points, seed_indices, remaining_points_indices)
    
    if only_clust:
        return clusters_indx
        
    feature = {}
    locations = ins.tasks.location
    feature[1] = ins.tasks.nr_tasks
    big_rectangle = enclosing_rectangle(locations)
    feature[2] = big_rectangle['area']
    feature[3] = big_rectangle['perimeter']
    feature[4] = hull.volume # area
    feature[5] = hull.area # perimeter
    feature[6] = big_rectangle['width']
    feature[7] = big_rectangle['height']
    list_of_cust_distances = [y for row in [ins.tasks.travel_matrix[i][i+1:] 
                              for i in range(1,ins.n)] for y in row] #only upper triangular matrix
    feature[8] = np.mean(list_of_cust_distances)
    feature[9] = 0
    feature[10] = 0
    feature[11] = 0
    feature[12] = np.mean( [distance(customer,big_rectangle['centroid']) for customer in locations] )
    customer_centroid = cust_centroid(locations)
    feature[13] = np.mean( [distance(customer,customer_centroid) for customer in locations] )
    feature[14] = 0
    feature[15] = variance_of_angles(locations, customer_centroid)
    feature[16] = variance_of_angles(locations, big_rectangle['centroid'] )
    feature[17] = np.var(np.array(locations))   ## ???????????
    feature[18] = np.var([p[0]*p[1] for p in locations])
    feature[19] = 0
    feature[20] = np.var( [distance(customer, customer_centroid) for customer in locations] )  
    feature[21] = np.var( [distance(customer, big_rectangle['centroid']) for customer in locations] )  
    feature[22] = np.var(list_of_cust_distances)
    feature[23] = 0
    feature[24] = 0
    feature[25] = nr_of_cust_within_circle(locations, 0.5, customer_centroid, hull)
    feature[26] = nr_of_cust_within_circle(locations, 0.75, customer_centroid, hull)
    feature[27] = nr_of_cust_within_circle(locations, 0.5, big_rectangle['centroid'], hull)
    feature[28] = nr_of_cust_within_circle(locations, 0.75, big_rectangle['centroid'], hull)
    subrectangles10 = subrectangles(locations, big_rectangle, 10)
    subrectangles15 = subrectangles(locations, big_rectangle, 15)
    feature[29] = 0
    feature[30] = 0
    feature[31] = subrectangles10['mean dist activ recs']
    feature[32] = subrectangles10['mean dist cust within recs']
    feature[33] = 0
    feature[34] = 0
    feature[35] = subrectangles15['mean dist activ recs']
    feature[36] = subrectangles15['mean dist cust within recs']
    feature[37] = 0
    feature[38] = 0
    feature[39] = 0
    feature[40] = 0
    feature[41] = ins.shifts.nr_shifts
    feature[42] = 0
    feature[43] = 0
    feature[44] = mean_distance_within_clusters
    feature[45] = mean_dist_between_centroids
    feature[46] = mean_distance_to_centroid
    feature[47] = 0
    feature[48] = 0
    
    return feature, clusters_indx