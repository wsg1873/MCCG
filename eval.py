import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time


def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL evaluation: supports eval through training code as well as independently
    policies_list should be a list of policies of all the agents;
    len(policies_list) = num agents
    """
    if env is None or master is None: # if any one of them is None, generate both of them
        master, env = setup_master(args, return_env=True)

    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)
    env.seed(seed)

    if ob_rms is not None:
        obs_mean, obs_std = ob_rms
    else:
        obs_mean = None
        obs_std = None
    master.load_models(policies_list)
    master.set_eval_mode()

    num_eval_episodes = args.num_eval_episodes
    all_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    # TODO: provide support for recurrent policies and mask
    recurrent_hidden_states = None
    mask = None

    # world.dists at the end of episode for simple_spread
    final_min_dists = []
    num_success = 0
    episode_length = 0
    unconnections = 0
    coverage_rate_sum = 0
    coverage_rate = 0
    start = time.time()

    for t in range(num_eval_episodes):
        obs = env.reset()
        obs_1 = [np.zeros(len(obs[0])) for i in range(len(obs))]
        obs_2 = [np.zeros(len(obs[0])) for i in range(len(obs))]
        obs_3 = [np.zeros(len(obs[0])) for i in range(len(obs))]
        obs_4 = [np.zeros(len(obs[0])) for i in range(len(obs))]

        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        episode_rewards = np.full(env.n, 0.0)
        episode_steps = 0
        # if render:
        if render and t ==408:

            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            env.render(attn=attn)
            
        while not np.all(done):
            actions = []
            with torch.no_grad():
                actions = master.eval_act(obs, obs_1, obs_2,  obs_3, obs_4, recurrent_hidden_states, mask)
                # print(time.time()-start)
                start = time.time()

                # actions = master.eval_act(obs, recurrent_hidden_states, mask)
            episode_steps += 1
            obs, reward, done, info = env.step(actions)
            obs = normalize_obs(obs, obs_mean, obs_std)
            obs_1 = obs
            obs_2 = obs_1
            obs_3 = obs_2
            obs_4 = obs_3

            episode_rewards += np.array(reward)
            # if render:
            if render and t ==408:

                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                env.render(attn=attn)
                # if args.record_video:
                # time.sleep(0.1)
                if args.record_video:
                    # print(attn)
                    input('Press enter to continue: ')

        per_step_rewards[t] = episode_rewards
        num_success += info['n'][0]['is_success']
        unconnections += info['n'][0]['unconnections']
        if info['n'][0]['unconnections'] > 0:
            print('*********************************')
            print('unconnections:',t)
        episode_length = (episode_length*t + info['n'][0]['world_steps'])/(t+1)
        coverage_rate_sum += info['n'][0]['coverage_rate_sum']/info['n'][0]['world_steps']

        coverage_rate += info['n'][0]['coverage_rate']

        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.world.min_dists)
        elif args.env_name == 'simple_formation' or args.env_name=='simple_line' or args.env_name=='simple_tag':
            final_min_dists.append(env.world.dists)


        if render:
            print("Ep {} | Success: {} |connections:{} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['n'][0]['is_success'],bool(1-info['n'][0]['unconnections']),
                per_step_rewards[t][0],info['n'][0]['world_steps']))
            # print('unconnections  {:d}'.format(info['n'][0]['unconnections']))

        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents  num_eval_episodes
        # if render and t ==408:

        # if args.record_video:
        #     # print(attn)
        #     input('Press enter to continue: ')
    print('unconnections  {:d}/{:d}'.format(unconnections, num_eval_episodes))
    print('coverage_rate  {:f}'.format(coverage_rate/num_eval_episodes))

    print('coverage_rate_sum  {:f}'.format(coverage_rate_sum/num_eval_episodes))


    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length, unconnections, coverage_rate_sum, coverage_rate


if __name__ == '__main__':
    args = get_args()
    args.name = 'simple_spread'
    args.num_good_agents = 5
    args.num_adversaries = 10
    args.render = True
    # args.render = False

    args.seed = 15864
    args.load_dir = 'marlsave/spread_static/ep0.pt'

    NA = [5,8,10,13,15]


    args.num_eval_episodes = 1000
    args.arena_size = 1.0
    if args.cuda:
        torch.cuda.set_device(args.cuda_num)#
        print(args.cuda_num)
    # for i in range(len(NA)):
    #     args.num_adversaries = NA[i]

    # args.load_dir = 'marlsave/spread_static5_1.0_continue/ep10000.pt'
    print(args.num_good_agents,  args.num_adversaries, args.load_dir)
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length, unconnections, coverage_rate_sum, coverage_rate = evaluate(args, args.seed, 
                    policies_list, ob_rms, args.render, render_attn=args.masking)
    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f})"
            .format(per_step_rewards.mean(0),num_success,args.num_eval_episodes,episode_length))
    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))
