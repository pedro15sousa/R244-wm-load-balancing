""" Various auxiliary utilities """
import math
import sys
from os.path import join, exists
from os import getpid
from time import sleep
from torch.multiprocessing import Process, Queue
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
import park
from models import MDRNNCell, VAE, Controller

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    10, 11, 64, 64, 64


################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index, tmp_dir, args, time_limit):
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
        r_gen = RolloutGenerator(args['logdir'], device, time_limit)
        print("Rollout generator process {} ready!".format(p_index))
        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


def set_slave_routines(num_workers, tmp_dir, pre_args, time_limit):
    """ Set slave routines.
    We cannot do this directly in a Colab notebook because of multiprocessing 
    in Jupyter kernels. We need to host that logic in a separate python file.
    """
    # define queues and set workers
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, 
                                            tmp_dir, pre_args, time_limit)).start()

    return p_queue, r_queue, e_queue


def hot_encode_action(action):
    # Create a zero tensor for one-hot encoding
    one_hot_actions = torch.zeros(ASIZE)
    # Fill with ones at the corresponding indices
    one_hot_actions[action] = 1
    return one_hot_actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def sample_discrete_policy(action_space, seq_len):
    """ Sample a random sequence of actions 
    :args action_space: gym action space (assuming it has a `n` attribute for the number of actions)
    :args seq_len: number of actions returned

    :returns: sequence of seq_len actions sampled uniformly at random from action_space
    """
    return np.array([action_space.sample() for _ in range(seq_len)])

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v1 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        # assert exists(vae_file) and exists(rnn_file),\
        #     "Either vae or mdrnn is untrained."

        rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})

        # Directly print the information
        print("Loading MDRNN at epoch {} with test loss {}".format(
            rnn_state['epoch'], rnn_state['precision']))

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = park.make('load_balance')
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """
        Get action and transition.

        Use the current observation as state, then obtain estimation for next
        hidden state using the MDRNN and compute the controller's corresponding action.

        :args obs: current observation torch tensor
        :args hidden: current hidden state torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden: torch tensor
        """
        state = obs
        action = self.controller.select_action(state, hidden[0], deterministic=False)
        action_encoded = hot_encode_action(action).unsqueeze(0).to(self.device)

        _, _, _, _, _, next_hidden = self.mdrnn(action_encoded, state, hidden)
        return action, next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array
s
        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        if hasattr(self.env, 'render'):
            self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0

        while True:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            sacalar_action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(sacalar_action)
            print("Reward: ", reward)

            if render:
                self.env.render()

            cumulative += reward
  
            if done or i > self.time_limit:
                # print(cumulative)
                # print(- cumulative)
                return - cumulative
            i += 1








































# """ Various auxiliary utilities """
# import math
# from os.path import join, exists
# import torch
# from torchvision import transforms
# import numpy as np
# import park
# from models import MDRNNCell, VAE, Controller

# # Hardcoded for now
# ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
#     10, 11, 64, 64, 64

# def hot_encode_action(action):
#     # Create a zero tensor for one-hot encoding
#     one_hot_actions = torch.zeros(ASIZE)
#     # Fill with ones at the corresponding indices
#     one_hot_actions[action] = 1
#     return one_hot_actions

# def save_checkpoint(state, is_best, filename, best_filename):
#     """ Save state in filename. Also save in best_filename if is_best. """
#     torch.save(state, filename)
#     if is_best:
#         torch.save(state, best_filename)

# def sample_discrete_policy(action_space, seq_len):
#     """ Sample a random sequence of actions 
#     :args action_space: gym action space (assuming it has a `n` attribute for the number of actions)
#     :args seq_len: number of actions returned

#     :returns: sequence of seq_len actions sampled uniformly at random from action_space
#     """
#     return np.array([action_space.sample() for _ in range(seq_len)])

# def flatten_parameters(params):
#     """ Flattening parameters.

#     :args params: generator of parameters (as returned by module.parameters())

#     :returns: flattened parameters (i.e. one tensor of dimension 1 with all
#         parameters concatenated)
#     """
#     return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

# def unflatten_parameters(params, example, device):
#     """ Unflatten parameters.

#     :args params: parameters as a single 1D np array
#     :args example: generator of parameters (as returned by module.parameters()),
#         used to reshape params
#     :args device: where to store unflattened parameters

#     :returns: unflattened parameters
#     """
#     params = torch.Tensor(params).to(device)
#     idx = 0
#     unflattened = []
#     for e_p in example:
#         unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
#         idx += e_p.numel()
#     return unflattened

# def load_parameters(params, controller):
#     """ Load flattened parameters into controller.

#     :args params: parameters as a single 1D np array
#     :args controller: module in which params is loaded
#     """
#     proto = next(controller.parameters())
#     params = unflatten_parameters(
#         params, controller.parameters(), proto.device)

#     for p, p_0 in zip(controller.parameters(), params):
#         p.data.copy_(p_0)

# class RolloutGenerator(object):
#     """ Utility to generate rollouts.

#     Encapsulate everything that is needed to generate rollouts in the TRUE ENV
#     using a controller with previously trained VAE and MDRNN.

#     :attr vae: VAE model loaded from mdir/vae
#     :attr mdrnn: MDRNN model loaded from mdir/mdrnn
#     :attr controller: Controller, either loaded from mdir/ctrl or randomly
#         initialized
#     :attr env: instance of the CarRacing-v1 gym environment
#     :attr device: device used to run VAE, MDRNN and Controller
#     :attr time_limit: rollouts have a maximum of time_limit timesteps
#     """
#     def __init__(self, mdir, device, time_limit):
#         """ Build vae, rnn, controller and environment. """
#         # Loading world model and vae
#         vae_file, rnn_file, ctrl_file = \
#             [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

#         assert exists(vae_file) and exists(rnn_file),\
#             "Either vae or mdrnn is untrained."

#         vae_state, rnn_state = [
#             torch.load(fname, map_location={'cuda:0': str(device)})
#             for fname in (vae_file, rnn_file)]

#         for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
#             print("Loading {} at epoch {} "
#                   "with test loss {}".format(
#                       m, s['epoch'], s['precision']))

#         self.vae = VAE(ASIZE+1, LSIZE).to(device)
#         self.vae.load_state_dict(vae_state['state_dict'])

#         self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
#         self.mdrnn.load_state_dict(
#             {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

#         self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

#         # load controller if it was previously saved
#         if exists(ctrl_file):
#             ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
#             print("Loading Controller with reward {}".format(
#                 ctrl_state['reward']))
#             self.controller.load_state_dict(ctrl_state['state_dict'])

#         self.env = park.make('load_balance')
#         self.device = device

#         self.time_limit = time_limit

#     def get_action_and_transition(self, obs, hidden):
#         """ Get action and transition.

#         Encode obs to latent using the VAE, then obtain estimation for next
#         latent and next hidden state using the MDRNN and compute the controller
#         corresponding action.

#         :args obs: current observation (1 x 3 x 64 x 64) torch tensor
#         :args hidden: current hidden state (1 x 256) torch tensor

#         :returns: (action, next_hidden)
#             - action: 1D np array
#             - next_hidden (1 x 256) torch tensor
#         """
#         _, latent_mu, _ = self.vae(obs)
#         state = (latent_mu, hidden[0])
#         action = self.controller.select_action(state, deterministic=True)
#         action_encoded = hot_encode_action(action).unsqueeze(0).to(self.device)

#         _, _, _, _, _, next_hidden = self.mdrnn(action_encoded, latent_mu, hidden)
#         return action, next_hidden

#     def rollout(self, params, render=False):
#         """ Execute a rollout and returns minus cumulative reward.

#         Load :params: into the controller and execute a single rollout. This
#         is the main API of this class.

#         :args params: parameters as a single 1D np array

#         :returns: minus cumulative reward
#         """
#         # copy params into the controller
#         if params is not None:
#             load_parameters(params, self.controller)

#         obs = self.env.reset()

#         # This first render is required !
#         if hasattr(self.env, 'render'):
#             self.env.render()

#         hidden = [
#             torch.zeros(1, RSIZE).to(self.device)
#             for _ in range(2)]

#         cumulative = 0
#         i = 0

#         while True:
#             obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
#             sacalar_action, hidden = self.get_action_and_transition(obs, hidden)
#             obs, reward, done, _ = self.env.step(sacalar_action)
#             print("Reward: ", reward)

#             if render:
#                 self.env.render()

#             cumulative += reward
  
#             if done or i > self.time_limit:
#                 # print(cumulative)
#                 # print(- cumulative)
#                 return - cumulative
#             i += 1