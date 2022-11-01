import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


horizon=100000
trials =10
players=['Player A','Player B','Player C','Player D','Player E']
colorMap = ['r','darkorange','seagreen','deepskyblue','mediumslateblue','deeppink']
#result_5 = np.load('./Results/'+'ceN='+'5'+'K=3'+'.npz')
#result_10 = np.load('./Results/'+'ceN='+'10'+'K=5'+'.npz')
#result_15 = np.load('./Results/'+'ceN='+'15'+'K=8'+'.npz')
#result_20 = np.load('./Results/'+'ceN='+'20'+'K=15'+'.npz')

result_5 = np.load('./Results/'+'N5'+'.npz')
result_10 = np.load('./Results/'+'N10'+'.npz')
result_15 = np.load('./Results/'+'N15'+'.npz')
result_20 = np.load('./Results/'+'N20'+'.npz')


regrets_ucb_5 = result_5['regretsucb']
rewarducb_5=result_5['rewarducb']
cu_reward_5=result_5['cu_reward']
av_regret_5=result_5['av_regret']
cu_unstable_5=result_5['cu_unstable']
av_unstable_5=result_5['av_unstable']
players_5 = np.argmax(regrets_ucb_5[:,horizon-1])

regrets_ucb_10 = result_10['regretsucb']
rewarducb_10=result_10['rewarducb']
cu_reward_10=result_10['cu_reward']
av_regret_10=result_10['av_regret']
cu_unstable_10=result_10['cu_unstable']
av_unstable_10=result_10['av_unstable']
players_10 = np.argmax(regrets_ucb_10[:,horizon-1])

regrets_ucb_15 = result_15['regretsucb']
rewarducb_15=result_15['rewarducb']
cu_reward_15=result_15['cu_reward']
av_regret_15=result_15['av_regret']
cu_unstable_15=result_15['cu_unstable']
av_unstable_15=result_15['av_unstable']
players_15 = np.argmax(regrets_ucb_15[:,horizon-1])

regrets_ucb_20 = result_20['regretsucb']
rewarducb_20=result_20['rewarducb']
cu_reward_20=result_20['cu_reward']
av_regret_20=result_20['av_regret']
cu_unstable_20=result_20['cu_unstable']
av_unstable_20=result_20['av_unstable']
players_20 = np.argmax(regrets_ucb_20[:,horizon-1])
plt.figure(dpi = 200)

plt.errorbar(range(horizon),regrets_ucb_5[players_5],fmt = '-', yerr=np.std(regrets_ucb_5[players_5])/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),regrets_ucb_10[players_10],fmt = '-', yerr=np.std(regrets_ucb_10[players_10])/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),regrets_ucb_15[players_15],fmt = '-', yerr=np.std(regrets_ucb_15[players_15])/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),regrets_ucb_20[players_20],fmt = '-', yerr=np.std(regrets_ucb_20[players_20])/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Cumulative Regret',fontsize = 16)
plt.title("Cumulative regret",fontsize = 16)
plt.savefig('./CAUCB_Cumulative_Regret_varying_N.pdf')
plt.close()


plt.figure(dpi = 200)

plt.errorbar(range(horizon),rewarducb_5[players_5],fmt = '-', yerr=np.std(rewarducb_5[players_5])/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),rewarducb_10[players_10],fmt = '-', yerr=np.std(rewarducb_10[players_10])/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),rewarducb_15[players_15],fmt = '-', yerr=np.std(rewarducb_15[players_15])/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),rewarducb_20[players_20],fmt = '-', yerr=np.std(rewarducb_20[players_20])/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Averaged Reward',fontsize = 16)
plt.title("Averaged Reward",fontsize = 16)
plt.savefig('./CAUCB_Averaged_Reward_varying_N.pdf')
plt.close()


plt.figure(dpi = 200)
plt.errorbar(range(horizon),av_regret_5[players_5],fmt = '-', yerr=np.std(av_regret_5[players_5])/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),av_regret_10[players_10],fmt = '-', yerr=np.std(av_regret_10[players_10])/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),av_regret_15[players_15],fmt = '-', yerr=np.std(av_regret_15[players_15])/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),av_regret_20[players_20],fmt = '-', yerr=np.std(av_regret_20[players_20])/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Averaged Regret',fontsize = 16)
plt.title("Averaged Regret",fontsize = 16)
plt.savefig('./CAUCB_Averaged_Regret_varying_N.pdf')
plt.close()

plt.figure(dpi = 200)
plt.errorbar(range(horizon),av_unstable_5,fmt = '-', yerr=np.std(av_unstable_5)/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),av_unstable_10,fmt = '-', yerr=np.std(av_unstable_10)/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),av_unstable_15,fmt = '-', yerr=np.std(av_unstable_15)/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),av_unstable_20,fmt = '-', yerr=np.std(av_unstable_20)/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Averaged Unstability',fontsize = 16)
plt.title("Averaged Unstability",fontsize = 16)
plt.savefig('./CAUCB_Averaged_Unstability_varying_N.pdf')
plt.close()

plt.figure(dpi = 200)
plt.errorbar(range(horizon),cu_unstable_5,fmt = '-', yerr=np.std(cu_unstable_5)/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),cu_unstable_10,fmt = '-', yerr=np.std(cu_unstable_10)/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),cu_unstable_15,fmt = '-', yerr=np.std(cu_unstable_15)/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),cu_unstable_20,fmt = '-', yerr=np.std(cu_unstable_20)/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Cumulative Unstability',fontsize = 16)
plt.title("Cumulative Unstability",fontsize = 16)
plt.savefig('./CAUCB_Cumulative_Unstability_varying_N.pdf')
plt.close()

plt.figure(dpi = 200)
plt.errorbar(range(horizon),cu_reward_5[players_5],fmt = '-', yerr=np.std(cu_reward_5[players_5])/np.sqrt(trials),color=colorMap[0], label='N=5', errorevery = 5000)
plt.errorbar(range(horizon),cu_reward_10[players_10],fmt = '-', yerr=np.std(cu_reward_10[players_10])/np.sqrt(trials),color=colorMap[1], label='N=10', errorevery = 5000)
plt.errorbar(range(horizon),cu_reward_15[players_15],fmt = '-', yerr=np.std(cu_reward_15[players_15])/np.sqrt(trials),color=colorMap[2], label='N=15', errorevery = 5000)
plt.errorbar(range(horizon),cu_reward_20[players_20],fmt = '-', yerr=np.std(cu_reward_20[players_20])/np.sqrt(trials),color=colorMap[3], label='N=20', errorevery = 5000)
        

        
       
plt.legend()
plt.xlabel('Time',fontsize = 16)
plt.ylabel('Expected Cumulative Reward',fontsize = 16)
plt.title("Cumulative Reward",fontsize=16)

plt.savefig('./CAUCB_Cumulative_Reward_varying_N.pdf')
plt.close()