import os
import json
import datetime
import numpy as np
import torch
import utils
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter
from eval import evaluate
from learner import setup_master
from pprint import pprint
import logging

cuda_num= 2

np.set_printoptions(suppress=True, precision=4)


def train(args, return_early=False):
    writer = SummaryWriter(args.log_dir)    
    envs = utils.make_parallel_envs(args) #
    master = setup_master(args) #  learner.py
    # used during evaluation onlyq
    eval_master, eval_env = setup_master(args, return_env=True) 
    obs = envs.reset() # shape - num_processes x num_agents x obs_dim observation
    master.initialize_obs(obs)
    n = len(master.all_agents)#
    episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    for j in range(args.num_updates):
        # print(j)
        for step in range(args.num_steps):#128
            # print(step)
            with torch.no_grad():
                actions_list = master.act(step)#list(num_agents) (num_agents,num_process)#list num-agnets num_process 1 :3X32
            agent_actions = np.transpose(np.array(actions_list),(1,0,2)) #agent_actions num_process num-agnets 1 ndarray(32,3,1)
            obs, reward, done, info = envs.step(agent_actions) # obs :ndarry(32,3,10),reward：ndarry(32,3)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device) # tensor,(32,3)
            episode_rewards += reward
            masks = torch.FloatTensor(1-1.0*done).to(args.device)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            master.update_rollout(obs, reward, masks)
          
        master.wrap_horizon()
        return_vals = master.update() #ndarray(3,3)
        value_loss = return_vals[:, 0]
        action_loss = return_vals[:, 1]
        dist_entropy = return_vals[:, 2]
        master.after_update()

        if j%args.save_interval == 0 and not args.test:#200，
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir+'/ep'+str(j)+'.pt'
            torch.save(savedict, savedir)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j%args.log_interval == 0:
            end = datetime.datetime.now()
            seconds = (end-start).total_seconds()
            mean_reward = final_rewards.mean(dim=0).cpu().numpy()
            logging.info("Updates {} | Num timesteps {} | Time {} | FPS {}\nMean reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
                  format(j, total_num_steps, str(end-start), int(total_num_steps / seconds),
                  mean_reward, dist_entropy[0], value_loss[0], action_loss[0]))
            # print("Updates {} | Num timesteps {} | Time {} | FPS {}\nMean reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
            #       format(j, total_num_steps, str(end-start), int(total_num_steps / seconds),
            #       mean_reward, dist_entropy[0], value_loss[0], action_loss[0]))
            if not args.test:
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/training_reward', mean_reward[idx], j)

                writer.add_scalar('all/value_loss', value_loss[0], j)
                writer.add_scalar('all/action_loss', action_loss[0], j)
                writer.add_scalar('all/dist_entropy', dist_entropy[0], j)

        if args.eval_interval is not None and j%args.eval_interval==0:
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            save_dir = str(args.save_dir)
            env_name = str(args.env_name)
            logging.info('===========================================================================================')
            logging.info('***************Info*************')
            logging.info('savedir{:s}'.format(save_dir))
            logging.info('cuda_num{:s}'.format(str(cuda_num)))
            logging.info('env_name:{:s}  ,  size: {:s} , masking is {:s}'.format(env_name,str(args.arena_size),str(args.masking)))#'masking is{:d}'.format(args.masking) , 'size{:s}'.format(str(args.arena_size))
            logging.info('version:  coverage_reward -1 + 1*connection_reward')
            logging.info('********************************')
            _, eval_perstep_rewards, final_min_dists, num_success, eval_episode_len, unconnections, coverage_rate_sum, coverage_rate = evaluate(args, None, master.all_policies,
                                                                                               ob_rms=ob_rms, env=eval_env,
                                                                                               master=eval_master)
            logging.info('Evaluation {:d} | Mean per-step reward {:.2f}'.format(j//args.eval_interval, eval_perstep_rewards.mean()))
            logging.info('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
            logging.info('unconnections {:d}/{:d}'.format(unconnections, args.num_eval_episodes))
            logging.info('coverage_rate {:f}/{:d}'.format(coverage_rate, args.num_eval_episodes))
            logging.info('coverage_rate_sum {:f}/{:d}'.format(coverage_rate_sum, args.num_eval_episodes))

            if final_min_dists and args.env_name != 'simple_tag':
                logging.info('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
                logging.info('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
            logging.info('===========================================================================================\n')

            if not args.test:
                writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, j)
                writer.add_scalar('all/episode_length', eval_episode_len, j)
                writer.add_scalar('all/unconnections', unconnections, j)
                writer.add_scalar('all/coverage_rate', coverage_rate, j)
                writer.add_scalar('all/coverage_rate_sum', coverage_rate_sum, j)

                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], j)
                    if final_min_dists and args.env_name != 'simple_tag':
                        writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], j)

            curriculum_success_thres = 0.9
            if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
                savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
                ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                savedict['ob_rms'] = ob_rms
                savedir = args.save_dir+'/ep'+str(j)+'.pt'
                torch.save(savedict, savedir)
                print('===========================================================================================\n')
                print('{} agents: training complete. Breaking.\n'.format(args.num_good_agents + args.num_adversaries))
                print('===========================================================================================\n')
                break

    writer.close()
    if return_early:
        return savedir

if __name__ == '__main__':
    args = get_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  #
    rq = 'outputlog'
    log_path = args.save_dir+'/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  #
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  #


    formatter = logging.Formatter("%(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    if args.cuda:
        torch.cuda.set_device(args.cuda_num)#
        print(cuda_num)
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes#//50e6//128//32
    torch.manual_seed(args.seed)##
    torch.set_num_threads(1)
    print(torch.get_num_threads())

    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    train(args)
