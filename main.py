import gym
import numpy as np
import agents as Agents
from utils import plot_learning_curve, plot_learning_curve_line, make_env, write_file
from gym import wrappers
import argparse, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Learning from DRL')
    parser.add_argument('-n_games', type=int, default=20,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.00025,
                        help='Learning rate for optimizer: pong, boxing, bowling: 0.0001, freeway: 0.00025')
    parser.add_argument('-eps_min', type=float, default=0.1,
            help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                                    help='Discount factor for update equation. pong, boxing, freeway: 0.99 ')
    parser.add_argument('-eps_dec', type=float, default=1e6,
                        help='Linear factor for decreasing epsilon: pong: 1e-5, boxing: 0.995 - DuelingDQN-lr 0.0001-eps_min 0.001 (DDDQN0.01), freeway: 1e6')
    parser.add_argument('-eps', type=float, default=1.0,
        help='Starting value for epsilon in epsilon-greedy action selection')
    
    parser.add_argument('-max_mem', type=int, default=50000, #~13Gb
                                help='Maximum size for memory replay buffer')
    parser.add_argument('-repeat', type=int, default=4,
                            help='Number of frames to repeat & stack')
    parser.add_argument('-bs', type=int, default=32,
                            help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=1000,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='FreewayDeterministic-v4',
                            help='Atari environment.\nPongNoFrameskip-v4\n BoxingNoFrameskip-v4\n\
                                  BreakoutNoFrameskip-v4\n \
                                  SpaceInvadersNoFrameskip-v4\n \
                                  EnduroNoFrameskip-v4\n \
                                  AtlantisNoFrameskip-v4')
    
    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                    help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-clip_rewards', type=bool, default=False,
                        help='Clip rewards to range -1 to 1')
    parser.add_argument('-no_ops', type=int, default=0,
                        help='Max number of no ops for testing')
    
    args = parser.parse_args()
    ###############################################################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    env = make_env(env_name=args.env, repeat=args.repeat,
                  clip_rewards=args.clip_rewards, no_ops=args.no_ops)
    ###############################################################
  
    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    print(agent_)
    load_checkpoint = False
    n_games = args.n_games

    agent = agent_(gamma=args.gamma,
                epsilon=args.eps,
                lr=args.lr,
                input_dims=env.observation_space.shape,
                n_actions=env.action_space.n,
                mem_size=args.max_mem,
                eps_min=args.eps_min,
                batch_size=args.bs,
                replace=args.replace,
                eps_dec=args.eps_dec,
                chkpt_dir=args.path,
                algo=args.algo,
                env_name=args.env)
    
    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
            
    figure_file = 'plots/' + fname + '.png'
    figure_file1 = 'plots/eps_' + fname + '.png'
    figure_file2 = 'plots/reward_' + fname + '.png'
    figure_file3 = 'plots/avg_reward_' + fname + '.png'
    _file = 'plots/' + fname + '.txt'
    
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    #and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array, avg_scores = [], [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done,_, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()
                
            observation = observation_
            n_steps += 1
            
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print('Episode: ', i,'\t Score: ', score,
             '\tAverage score: %.1f' % avg_score, '\tBest score: %.2f' % best_score,
            '\tEpsilon: %.2f' % agent.epsilon, '\tSteps:', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, avg_scores, figure_file, None, 20)
    plot_learning_curve_line(steps_array, scores, eps_history, avg_scores, figure_file1, figure_file2,figure_file3, None, 20)
    write_file(steps_array, scores, eps_history, avg_scores, _file, 20)
