"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain '
                    'return estimate.')
parser.add_argument('--pop-size', type=int, help='Population size.')
parser.add_argument('--target-return', type=float, help='Stops once the return '
                    'gets above target_return')
parser.add_argument('--display', action='store_true', help="Use progress bars if "
                    "specified.")
parser.add_argument('--max-workers', type=int, help='Maximum number of workers.',
                    default=32)
args = parser.parse_args()

# Max number of workers. M

# multiprocessing variables
n_samples = args.n_samples
pop_size = args.pop_size
num_workers = min(args.max_workers, n_samples * pop_size)
time_limit = 10000

# create tmp dir if non existent and clean it if existent
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# create ctrl dir if non exitent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)


################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    # gpu = p_index % torch.cuda.device_count()
    # device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    # Check if CUDA-capable GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = p_index % torch.cuda.device_count()
        device = torch.device('cuda:{}'.format(gpu))
    else:
        # Use CPU if no GPUs are available
        device = torch.device('cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a') 

    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, device, time_limit)
        print("Rollout generator process {} ready!".format(p_index))
        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    print("Evaluation finished: ", np.mean(restimates))
    return best_guess, np.mean(restimates), np.std(restimates)


# ################################################################################
# #                Define queues and start workers                               #
# ################################################################################
# p_queue = Queue()
# r_queue = Queue()
# e_queue = Queue()

# for p_index in range(num_workers):
#     Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()


################################################################################
#                           Launch CMA                                         #
################################################################################
if __name__ == '__main__':
    # define queues and set workers
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()
    
    # Initialize lists to store total actions and variance for plotting
    action_history = []
    variance_history = []
    reward_history = []

    controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance

    # define current best and load parameters
    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                {'popsize': pop_size})

    epoch = 0
    log_step = 2
    training_rewards = []
    training_variances = []
    total_actions = 0
    n_actions_per_sample = 1000

    # p_queue, r_queue, e_queue = set_queues(num_workers)

    while not es.stop():
        print("\n********** Generation {} ************".format(epoch))
        if cur_best is not None and - cur_best > args.target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  # result list
        solutions = es.ask()

        batch_actions = 0
        batch_rewards = []

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))
    
        # retrieve results
        if args.display:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            batch_rewards.append(r)
            if args.display:
                pbar.update(1)
        if args.display:
            pbar.close()
        
        batch_actions = pop_size * n_samples * n_actions_per_sample  # Compute total actions for the batch
        total_actions += batch_actions  # Update the total actions
        action_history.append(total_actions)  # Store the action history for plotting

        # Calculate and store variance
        batch_variance = np.var(batch_rewards)
        variance_history.append(batch_variance)

        # Store average reward for plotting
        avg_reward = -np.mean(r_list)
        print("Average reward: ", avg_reward)
        reward_history.append(avg_reward)

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list)
            print("Current evaluation: {}".format(-best))
            training_rewards.append(-best)
            training_variances.append(std_best)
            print("Lenght of training rewards: ", len(training_rewards))
            print("Lenght of action history: ", len(action_history))
            print("Lenght of variance history: ", len(variance_history))
            print("Lenght of reward history: ", len(reward_history))
            if cur_best != None: print("Current overall best: {}".format(-cur_best))
            if not cur_best or best < cur_best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                    'reward': - cur_best,
                    'state_dict': controller.state_dict(),
                    'training_rewards': training_rewards,
                    'training_variances': training_variances,
                    'action_history': action_history,
                    'variance_history': variance_history,
                    'reward_history': reward_history},
                    join(ctrl_dir, 'best.tar'))
            if - best > args.target_return:
                print("Terminating controller training with value {}...".format(-best))
                break


        epoch += 1

    es.result_pretty()
    e_queue.put('EOP')
