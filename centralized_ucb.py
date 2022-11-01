import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

#from src.arm import Arm




class Centralized_UCB(object):

    def __init__(self):
        self.num_players = 10
        self.num_arms = 5
        
        self.horizon = 500000
        self.trials = 10

        #self.players_ranking = [ np.random.permutation(self.num_arms).tolist() for j in range(self.num_players)]
        #self.arms_rankings = [ np.random.permutation(self.num_players).tolist() for j in range(self.num_arms)]
        #self.players_mean = [np.linspace(0.9, 0, self.num_arms) for j in range(self.num_players)]

        
        market = np.load('./Markets/beta_'+str(0)+'N_'+str(self.num_players)+'K_'+str(self.num_arms)+'.npz')
        self.players_ranking = market['player_rank'].tolist()
        self.arms_rankings = market['arm_rank'].tolist()
        self.players_mean = market['player_mean'].tolist()
        self.arms_capacity = [2,2,2,2,2]
        #self.arms_capacity = [3,2,2,3,2,2,2,2,1,1]
        self.max_cap = 2
        self.p_lambda = 0.1
        self.epsilon = 10**(-10)

        self.pessimal_matching = self.get_pessimal_matching(self.players_ranking,self.arms_rankings).tolist()
        print(self.pessimal_matching)
        #self.pessimal_matching = [0,0,1,1,2]


    def isUnstable(self, player_matching):
        # arm_matching: [0,1,-1]
        # arm 0 matches player 0; arm 1 matches player 1; arm 2 matches nothing

        # if unstable return 1, otherwise return 0
        player_matching = player_matching.tolist()

        if -1 in player_matching:
            return 1
        arm_matching = [[] for j in range(self.num_arms)]
        for p_idx in range(self.num_players):
            if player_matching[p_idx] != -1:
                arm_matching[int(player_matching[p_idx])].append(p_idx)
        
        for a_idx in range(self.num_arms):
            if len(arm_matching[a_idx])==0:
                return 1
        # find blocking pair
        for p_idx in range(self.num_players):
            for possible_arm_rank in range(self.players_ranking[p_idx].index(player_matching[p_idx])):
                arm = self.players_ranking[p_idx][possible_arm_rank]
                for j in range(len(arm_matching[arm])):
                    for possible_player_rank in range(self.arms_rankings[arm].index(arm_matching[arm][j])):
                        if self.arms_rankings[arm][possible_player_rank] == p_idx:
                            return 1
        return 0


    def get_pessimal_matching(self,players_rankings,arms_rankings):
        # propose_order records the order arms should follow while proposing
        #init_propose_order = np.zeros(self.num_arms, int)
        #propose_order = init_propose_order
        # matched record whether a specific player is matched or not
        matched = np.zeros(self.num_arms)
        # matching records the choice of a player for a specific arm
        matching = [[] for _ in range(self.num_players)]
        reject = np.zeros((self.num_players,self.num_arms))

        # Terminates if all matched
        while np.sum(matched) != self.num_players:
            matched[:] = 0
            proposed = [[] for _ in range(self.num_players)]    
            # arms propose at the same time
            for a_idx in range(self.num_arms):
                
                # p_proposal is the index of an arm
                # propose_order is the vector, p_o[i] is the order of player i's next proposal
                propose_count = 0
                for p_idx in arms_rankings[a_idx]:
                    if reject[p_idx][a_idx] == 0:
                        proposed[p_idx].append(a_idx)
                        propose_count += 1
                        if propose_count == self.arms_capacity[a_idx]:
                            break

            # arms choose its player
            for p_idx in range(self.num_players):
                p_choices = proposed[p_idx]

                if len(p_choices) != 0:    
                    # each arm chooses the its most preferable one
                    p_choice = next((x for x in players_rankings[p_idx] if x in proposed[p_idx]), None)
                    # update arm's choice where there should only be one left
                    matching[p_idx] = [p_choice]
                    # update player's state of matched
                    for a_idx in p_choices:
                        if a_idx != p_choice:
                            reject[p_idx][a_idx] = 1
                        else:
                            matched[a_idx] += 1
                        
        return np.squeeze(matching)

    def SADA(self,player_ranking):
        # propose_order records the order players should follow while proposing
            init_propose_order = np.zeros(self.num_players, int)
            propose_order = init_propose_order
            # matched record whether a specific player is matched or not
            matched = np.zeros(self.num_players, bool)
            # matching records the choice of a player for a specific arm
            matching = [[] for _ in range(self.num_arms)]
            matching_p = [[] for _ in range(self.num_players)]

            # Terminates if all matched
            while np.sum(matched) != self.num_players:
                matching = [[] for _ in range(self.num_arms)]
                # players propose at the same time
                for p_idx in range(self.num_players):
                    
                    # p_proposal is the index of an arm
                    # propose_order is the vector, p_o[i] is the order of player i's next proposal
                    p_proposal = player_ranking[p_idx][propose_order[p_idx]]
                    matching[p_proposal].append(p_idx)

                # arms choose its player
                for a_idx in range(self.num_arms):
                    a_choices = matching[a_idx]

                    if len(a_choices) != 0:    
                        # each arm chooses the its most preferable one
                        a_choice = []
                        it = iter([ x for x in self.arms_rankings[a_idx] if x in matching[a_idx]])
                        for i in range(min(len(matching[a_idx]),self.arms_capacity[a_idx])):
                            a_choice.append(next(it, None))
                        # update arm's choice where there should only be one left
                        matching[a_idx] = a_choice
                        # update player's state of matched
                        for p_idx in a_choices:
                            if p_idx in a_choice:
                                matched[p_idx] = 1
                                matching_p[p_idx] = a_idx
                            else:
                                matched[p_idx] = 0
                                propose_order[p_idx] = propose_order[p_idx] +1
                             
        
            return np.squeeze(matching_p)
                


    def run_UCB(self):
        cumulative_regrets = np.zeros([self.num_players, self.horizon])
        averaged_rewards = np.zeros([self.num_players, self.horizon])
        averaged_regrets = np.zeros([self.num_players, self.horizon])
        cumulative_rewards = np.zeros([self.num_players, self.horizon])
        averaged_unstable = np.zeros( self.horizon)
        cumulative_unstable = np.zeros(self.horizon)
        ucb = np.zeros([self.num_players,self.num_arms,self.horizon])
        # # Using to draw figure
        # cumulative_regrets = np.zeros([self.num_players, self.trials, self.horizon])
        # averaged_regrets = np.zeros([self.num_players, self.trials, self.horizon])
        # cumulative_unstable = np.zeros([self.trials, self.horizon])
        # averaged_unstable = np.zeros([self.trials, self.horizon])

        # Using to save data
        regrets_trials = np.zeros([self.num_players, self.trials, self.horizon])
        rewards_trials = np.zeros([self.num_players, self.trials, self.horizon])
        unstable_trials = np.zeros([self.trials, self.horizon])

        
        for trial in tqdm(range(self.trials), ascii=True, desc="Running the Centralized-"+'UCB'):
            averaged_rewards_one_trial = np.zeros([self.num_players, self.horizon])
            averaged_regrets_one_trial = np.zeros([self.num_players, self.horizon])
            unstable_one_trial = np.zeros(self.horizon)
            averaged_unstable_one_trial = np.zeros(self.horizon)
            regrets_one_trial = np.zeros([self.num_players, self.horizon])
            rewards_one_trial = np.zeros([self.num_players, self.horizon])
            ucb_one_trial = np.zeros([self.num_players,self.num_arms,self.horizon])
            

           
            self.players_es_mean = [np.zeros(self.num_arms) for j in range(self.num_players)]
            self.players_count = [np.zeros(self.num_arms) for j in range(self.num_players)]
            self.players_ucb = [np.ones(self.num_arms) * np.inf for j in range(self.num_players)]
            
            
            for round in range(self.horizon):
                # print("----------------TS time step: ",round, "-------------")

                current_player_ranking = [ np.zeros(self.num_arms) for j in range(self.num_players)]
                
                # print("Theta",self.players_theta)
               
                for j in range(self.num_players):
                    for a in range(self.num_arms):
                        self.players_ucb[j][a] = self.players_es_mean[j][a]+np.sqrt(3 * np.log(round+1) / (2*(self.players_count[j][a] + self.epsilon)))
                        ucb_one_trial[j][a][round] = self.players_ucb[j][a]
                    current_player_ranking[j] = np.argsort(-self.players_ucb[j])
                
                
                current_match = self.SADA(current_player_ranking)

                for p_idx, a_idx in enumerate(current_match):
                    #print(current_match)
                    reward = np.random.binomial(1, self.players_mean[p_idx][a_idx])
                    
                    self.players_count[p_idx][a_idx]+=1
                    self.players_es_mean[p_idx][a_idx]+= (reward-self.players_es_mean[p_idx][a_idx]) / self.players_count[p_idx][a_idx]
                        
                    
                    regrets_one_trial[p_idx][round]=max(0,self.players_mean[p_idx][self.pessimal_matching[p_idx]] - self.players_mean[p_idx][a_idx])
                    rewards_one_trial[p_idx][round] = self.players_mean[p_idx][a_idx]
                    # averaged_rewards_one_trial[p_idx][round]= mean(rewards_one_trial[p_idx][0:round+1])
                    # averaged_regrets_one_trial[p_idx][round]= mean(regrets_one_trial[p_idx][0:round+1])
                    averaged_rewards_one_trial[p_idx][round] = (averaged_rewards_one_trial[p_idx][round-1] * round + rewards_one_trial[p_idx][round])/(round+1)
                    averaged_regrets_one_trial[p_idx][round] = (averaged_regrets_one_trial[p_idx][round-1] * round + regrets_one_trial[p_idx][round])/(round+1)
                        
                    
                        
                # Here: whether stable matching according to last_pulled.
                unstable_one_trial[round] = self.isUnstable(current_match)
                # averaged_unstable_one_trial[round] = mean(unstable_one_trial[0:round+1])
                averaged_unstable_one_trial[round] = (averaged_unstable_one_trial[round-1]*round + unstable_one_trial[round])/(round+1)

                if round==self.horizon-1:
                    print('--------'+'UCB'+': Arm matching = ',current_match)    
                    


            # Save Data
            for i in range(self.num_players):
                # cumulative_regrets[i][trial]=np.cumsum(np.array(regrets_one_trial[i]), axis=0)
                # averaged_regrets[i][trial]=cumulative_regrets[i][trial]/range(1,self.horizon+1)

                regrets_trials[i][trial] = regrets_one_trial[i]
                rewards_trials[i][trial] = rewards_one_trial[i]

            #unstable_trials[trial] = unstable_one_trial
            cumulative_regrets += np.cumsum(np.array(regrets_one_trial), axis=1)
            cumulative_rewards += np.cumsum(np.array(rewards_one_trial), axis=1)
            averaged_rewards += averaged_rewards_one_trial
            averaged_regrets += averaged_regrets_one_trial
            cumulative_unstable += np.cumsum(np.array(unstable_one_trial))
            averaged_unstable += averaged_unstable_one_trial
            ucb += ucb_one_trial

            # cumulative_unstable[trial] = np.cumsum(np.array(unstable_one_trial), axis=0)
            # averaged_unstable[trial] = cumulative_unstable[trial]/range(1,self.horizon+1)

            # cumulative_regrets = np.array(cumulative_regrets)
            # averaged_regrets = np.array(averaged_regrets)
            # cumulative_unstable = np.array(cumulative_unstable)
            # averaged_unstable = np.array(averaged_unstable)

            # print(alg, "Averaged_regret",averaged_regrets)
            # print(alg, "Averaged_unstable",averaged_unstable)
        cumulative_regrets /= self.trials
        cumulative_rewards /= self.trials
        averaged_rewards /= self.trials
        averaged_regrets /= self.trials
        cumulative_unstable /= self.trials
        averaged_unstable /=self.trials
        ucb /= self.trials
        
        
       
        
        return cumulative_regrets,averaged_rewards,cumulative_rewards,averaged_regrets,cumulative_unstable,averaged_unstable,ucb


    
    # regret_list: dimention: alg, player, trials, horizon
    def plot_regret_allPlayers_for_difAlg(alg_list,player_start, player_end, regret_list, horizon, trials, ylabel, title, path):
            matplotlib.rcParams.update({'figure.autolayout': True})
            plt.rc('font', size=12)          # controls default text sizes
            plt.rc('axes', titlesize=12)     # fontsize of the axes title
            plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
            plt.rc('legend', fontsize=12)    # legend fontsize
            plt.rc('figure', titlesize=13)  # fontsize of the figure title

            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

            fig, ax = plt.subplots()
            errorevery = 5000

            fmt_map = ['-','--','-.',':']

            players=['Player A','Player B','Player C','Player D','Player E']
            colorMap = ['r','darkorange','seagreen','deepskyblue','mediumslateblue','deeppink']
            
            for alg in range(len(alg_list)):
                for p_idx in range(player_start,player_end):
                    regret_mean = np.mean(regret_list[alg][p_idx], axis=0)
                    plt.errorbar(range(horizon), regret_mean, fmt=fmt_map[alg], yerr=np.std(regret_mean)/np.sqrt(trials), color=colorMap[p_idx], label=alg_list[alg]+', '+players[p_idx], errorevery = errorevery)

            
            plt.locator_params('x',nbins=6)
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel(ylabel)

            
            plt.title(title)
        
            plt.savefig(path)
            plt.close(fig)

    def plot_centralized(self,regrets_ucb,rewards_ucb,cu_reward,av_regret,cu_unstable,av_unstable):
        players=['Player A','Player B','Player C','Player D','Player E']
        colorMap = ['r','darkorange','seagreen','deepskyblue','mediumslateblue','deeppink']
        plt.figure(dpi = 200)
        #regret_mean = np.zeros(self.num_players)
        #for p_idx in range(self.num_players):
        #    regret_mean[p_idx] = np.mean(regrets_ucb[p_idx])
        plt.errorbar(range(self.horizon),regrets_ucb[0],fmt = '-', yerr=np.std(regrets_ucb[0])/np.sqrt(self.trials),color=colorMap[0], label=players[0], errorevery = 5000)
        
        plt.errorbar(range(self.horizon),regrets_ucb[2],fmt = '--', yerr=np.std(regrets_ucb[2])/np.sqrt(self.trials),color=colorMap[2], label=players[2], errorevery = 5000)
        plt.errorbar(range(self.horizon),regrets_ucb[4],fmt = ':', yerr=np.std(regrets_ucb[4])/np.sqrt(self.trials),color=colorMap[4], label=players[4], errorevery = 5000)
        
       
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Cumulative Regret')
        plt.title("Cumulative regret")
        plt.savefig('./Cumulative_Regret_centralized_ucb.png')
        plt.close()


        plt.figure(dpi = 200)
        #regret_mean = np.zeros(self.num_players)
        #for p_idx in range(self.num_players):
        #    regret_mean[p_idx] = np.mean(regrets_ucb[p_idx])
        plt.errorbar(range(self.horizon),av_regret[0],fmt = '-', yerr=np.std(av_regret[0])/np.sqrt(self.trials),color=colorMap[0], label=players[0], errorevery = 5000)
        
        plt.errorbar(range(self.horizon),av_regret[2],fmt = '--', yerr=np.std(av_regret[2])/np.sqrt(self.trials),color=colorMap[2], label=players[2], errorevery = 5000)
        plt.errorbar(range(self.horizon),av_regret[4],fmt = ':', yerr=np.std(av_regret[4])/np.sqrt(self.trials),color=colorMap[4], label=players[4], errorevery = 5000)
        
       
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Averaged Regret')
        plt.title("Averaged regret")
        plt.savefig('./Averaged_Regret_centralized_ucb.png')
        plt.close()

        plt.figure(dpi = 200)
        #regret_mean = np.zeros(self.num_players)
        #for p_idx in range(self.num_players):
        #    regret_mean[p_idx] = np.mean(regrets_ucb[p_idx])
        plt.errorbar(range(self.horizon),cu_reward[0],fmt = '-', yerr=np.std(cu_reward[0])/np.sqrt(self.trials),color=colorMap[0], label=players[0], errorevery = 5000)
        
        plt.errorbar(range(self.horizon),cu_reward[2],fmt = '--', yerr=np.std(cu_reward[2])/np.sqrt(self.trials),color=colorMap[2], label=players[2], errorevery = 5000)
        plt.errorbar(range(self.horizon),cu_reward[4],fmt = ':', yerr=np.std(cu_reward[4])/np.sqrt(self.trials),color=colorMap[4], label=players[4], errorevery = 5000)
        
       
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Cumulative Reward')
        plt.title("Cumulative rewards")
        plt.savefig('./Cumulative_Rewards_centralized_ucb.png')
        plt.close()

        plt.figure(dpi = 200)
        plt.errorbar(range(self.horizon),rewards_ucb[0],fmt = '-', yerr=np.std(rewards_ucb[0])/np.sqrt(self.trials),color=colorMap[0], label=players[0], errorevery = 5000)
        
        plt.errorbar(range(self.horizon),rewards_ucb[2],fmt = '--', yerr=np.std(rewards_ucb[2])/np.sqrt(self.trials),color=colorMap[2], label=players[2], errorevery = 5000)
        plt.errorbar(range(self.horizon),rewards_ucb[4],fmt = ':', yerr=np.std(rewards_ucb[4])/np.sqrt(self.trials),color=colorMap[4], label=players[4], errorevery = 5000)
        
     
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Averaged Reward')
        plt.title("Averaged reward")
        plt.savefig('./Averaged_Reward_centralized_ucb.png')
        

        plt.figure(dpi = 200)
        plt.errorbar(range(self.horizon),cu_unstable,fmt = '-', yerr=np.std(cu_unstable)/np.sqrt(self.trials),color=colorMap[0],  errorevery = 5000)
        
     
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Cumulative Unstablility')
        plt.title("Cumulative unstablility")
        plt.savefig('./Cumulative_unstability_centralized_ucb.png')

        plt.figure(dpi = 200)
        plt.errorbar(range(self.horizon),av_unstable,fmt = '-', yerr=np.std(av_unstable)/np.sqrt(self.trials),color=colorMap[0],  errorevery = 5000)
        
     
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Averaged Unstablility')
        plt.title("Averaged unstablility")
        plt.savefig('./Averaged_unstability_centralized_ucb.png')
        

        # regret_list: dimention: alg, player, trials, horizon

test = Centralized_UCB()
#regretsts,rewardts = test.run_CA_TS()
regretsucb,rewarducb,cu_reward,av_regret,cu_unstable,av_unstable,ucb = test.run_UCB()
np.savez('./Results/'+'ce'+'N=10'+'K='+'5'+'.npz', regretsucb=regretsucb,rewarducb=rewarducb,cu_reward=cu_reward,av_regret=av_regret,cu_unstable=cu_unstable,av_unstable=av_unstable)
np.savez('./Results/'+'ce'+'N=10'+'K='+'5'+'ucb'+'.npz', ucb=ucb)


test.plot_centralized(regretsucb,rewarducb,cu_reward,av_regret,cu_unstable,av_unstable)
#test.plot_regret_allPlayers_for_difAlg()





# arm_rank = [ [1,2,0], [0,1,2],[2,0,1] ]
# print(np.mean(arm_rank))
# print(type(np.mean(arm_rank)))



# test = Centralized_UCB()
# player_rank = [ [0,1,2],[1,0,2],[2,0,1]  ]
# arm_rank = [ [1,2,0], [0,1,2],[2,0,1] ]
# print(test.get_pessimal_matching(arm_rank,player_rank).tolist())


