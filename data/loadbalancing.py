"""
Generating data from the LoadBalancing Park environment.
"""

import argparse
from os.path import join, exists
import park
import numpy as np

def generate_data(rollouts, data_dir):
    """ Generates data """
    assert exists(data_dir), "The directory {} does not exist".format(data_dir)

    env = park.make('load_balance')
    
    for i in range(rollouts):
        print("Rollout #{}".format(i))
        env.reset()
        
        a_rollout = []
        s_rollout = []
        r_rollout = []
        d_rollout = [] 

        t = 0

        while True:
            action = env.action_space.sample()
            a_rollout += [action]
            t += 1

            s, r, done, _ = env.step(action)
            
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir)