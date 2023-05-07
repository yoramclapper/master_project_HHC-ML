import multiprocessing
import gomea
import instance6 as instance # YC: instance6 not in modules
import pickle


def get_result(instance):

    return gomea.gomea_solve(instance, population=600)

 
if __name__ == '__main__':
    
    # YC: not sure why data is split in 3
    for j in range(3):
        dataset = {
            'instance' : [],
            "total_travel_time" : [],
            "total_waiting_time" : [],
            "total_overtime" : [],
            "total_idle_time" : [],
            "info": 'instance 6 used (all customers one tw length)'
            }
        N = 10
        for i in range(N):
            #initialize
            pool_size = 40 #number of processes you want to run in parallel
            print('pool size', pool_size)
            inputs = [instance.get_ins() for _ in range(pool_size)]
            pool = multiprocessing.Pool()
            pool = multiprocessing.Pool(processes=pool_size)
            #runs get_result in parallel and stores output in array
            print('start multi-process')
            outputs = pool.map(get_result, inputs)
            print('-------------------------------------------multi-process completed, i =',i, 'j = ', j)
            dataset['instance'].extend(outputs)
            dataset['total_travel_time'].extend( [o['distance'] for o in outputs] )
            dataset['total_waiting_time'].extend( [o['waiting_time'] for o in outputs] )
            dataset['total_overtime'].extend( [o['shift_overtime'] for o in outputs] )
            dataset['total_idle_time'].extend( [o['idle_time'] for o in outputs] )
            
            with open('dataset_vrp{}.pickle'.format(j), 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # from gom8 it has pop 600