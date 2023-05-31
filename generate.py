import multiprocessing
import gomea
#import instance6 as instance # YC: instance6 not in modules
import instance # YC: added instead of instance6
import pickle
import time


def get_result(instance):
    return gomea.gomea_solve(instance, population=500)

 
if __name__ == '__main__':
    ts = time.time()
    
    for i in range(20):
        print('\n')
        print('batch:',i+1)
        print('\n')
        ts_split = time.time()
    
        dataset = {
            'instance' : [],
            "total_travel_time" : [],
            "total_waiting_time" : [],
            "total_overtime" : [],
            "total_idle_time" : [],
            "info": 'instance 6 used (all customers one tw length)'
            }
        
        #initialize
        pool_size = 300 #number of instances to run
        num_proc = multiprocessing.cpu_count()
        print('pool size:', pool_size)
        print('processors:',num_proc-1)
        
        inputs = [instance.get_ins() for _ in range(pool_size)]
        
        print('start processing')
        pool = multiprocessing.Pool(processes=num_proc-1)
        outputs = pool.map(get_result, inputs)
        print('processing completed')
    
        dataset['instance'].extend(outputs)
        dataset['total_travel_time'].extend( [o['distance'] for o in outputs] )
        dataset['total_waiting_time'].extend( [o['waiting_time'] for o in outputs] )
        dataset['total_overtime'].extend( [o['shift_overtime'] for o in outputs] )
        dataset['total_idle_time'].extend( [o['idle_time'] for o in outputs] )
                
        with open('dataset_vrp{}.pickle'.format(str(99)+'-'+str(i+1)), 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
        print('split:',time.time()-ts_split)
            
    print('elapsed time:',time.time()-ts)