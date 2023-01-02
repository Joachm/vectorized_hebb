import concurrent.futures
import copy
import pickle
import gym
import numpy as np
from ES_classes import OpenES
from hebbian_neural_net import HebbianNet
from rollout import fitness
import pybullet_envs


ENV_NAME = 'AntBulletEnv-v0'

env = gym.make(ENV_NAME)
inp_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

ARCHITECTURE = [inp_size, 128,64, action_size]
EPOCHS = 2001
TASK_PER_IND = 1
EVAL_EVERY = 10
popsize = 512
cpus = 64


runs = [ 'd_', 'e_']
for run in runs:

    init_net = HebbianNet(ARCHITECTURE)

    init_params = init_net.get_params()

    print('trainable parameters:', len(init_params))

    
    with open('log_'+str(run)+'.txt', 'a') as outfile:
        outfile.write('trainable parameters: ' + str(len(init_params))+'\n')

    solver = OpenES(len(init_params), 
                    popsize=popsize, 
                    rank_fitness=True,
                    antithetic=True,
                    learning_rate=0.01,
                    learning_rate_decay=0.9999,
                    sigma_init=0.1,
                    sigma_decay=0.999,
                    learning_rate_limit=0.001,
                    sigma_limit=0.01)
    solver.set_mu(init_params)

    def worker_fn(params):
        mean = 0
        for epi in range(TASK_PER_IND):
            net = HebbianNet(ARCHITECTURE)
            net.set_params(params)
            mean += fitness(net, ENV_NAME)
        return mean/TASK_PER_IND


    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        solutions = solver.ask()
        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            fitlist = executor.map(worker_fn, [params for params in solutions])

        fitlist = list(fitlist)
        solver.tell(fitlist)

        fit_arr = np.array(fitlist)

        print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
        with open('log_'+str(run)+'.txt', 'a') as outfile:
            outfile.write('epoch: ' + str(epoch)
                    + ' mean: ' + str(np.mean(fitlist)) 
                    + ' best: ' + str(np.max(fitlist)) 
                    + ' worst: ' + str(np.min(fitlist)) 
                    + ' std.: '  +str(np.std(fitlist)) + '\n')

        pop_mean_curve[epoch] = fit_arr.mean()
        best_sol_curve[epoch] = fit_arr.max()

        if (epoch + 1) % EVAL_EVERY == 0:
            with concurrent.futures.ProcessPoolExecutor(64) as executor:
                evaluations = executor.map(worker_fn, [solver.current_param() for i in range(64)])
            evaluations = list(evaluations)
            with open('log_'+str(run)+'.txt', 'a') as outfile:
                outfile.write('EVAL:   '+ ' mean: ' + str(np.mean(evaluations)) 
                        + ' best: ' + str(np.max(evaluations)) 
                        + ' worst: ' + str(np.min(evaluations)) 
                        + ' std.: '  +str(np.std(evaluations)) + '\n')
            eval_curve[epoch] = np.mean(evaluations)

        #'''
        if (epoch + 1) % 500 == 0:
            print('saving..')
            pickle.dump((
                         solver,
                         copy.deepcopy(init_net),
                         pop_mean_curve,
                         best_sol_curve,
                         eval_curve
                         ), open(str(run)+'_' + str(len(init_params)) + '_' + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))

        #'''


