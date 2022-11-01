import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from scipy.signal import savgol_filter
plt.switch_backend('agg')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
horizon=100000
trials =10
players=['Player A','Player B','Player C','Player D','Player E']
colorMap = ['r','darkorange','seagreen','deepskyblue','mediumslateblue','deeppink']
#result_5 = np.load('./Results/'+'ceN='+'5'+'K=3'+'.npz')
#result_10 = np.load('./Results/'+'ceN='+'10'+'K=5'+'.npz')
#result_15 = np.load('./Results/'+'ceN='+'15'+'K=8'+'.npz')
#result_20 = np.load('./Results/'+'ceN='+'20'+'K=15'+'.npz'
result_etc_K_2 = np.load('./Results/'+'etc'+'N=15'+'K=2'+'h=30' +'.npz')
result_etc_K_5 = np.load('./Results/'+'etc'+'N=15'+'K=5'+'h=100' +'.npz')
result_etc_K_10 = np.load('./Results/'+'etc'+'N=15'+'K=10'+'h=200' +'.npz')
result_etc_K_15 = np.load('./Results/'+'etc'+'N=15'+'K=15'+'h=300' +'.npz')

result_ucb_K_2 = np.load('./Results/'+'ce'+'N=15'+'K=2'+'.npz')
result_ucb_K_5 = np.load('./Results/'+'ce'+'N=15'+'K=5'+'.npz')
result_ucb_K_10 = np.load('./Results/'+'ce'+'N=15'+'K=10'+'.npz')
result_ucb_K_15 = np.load('./Results/'+'ce'+'N=15'+'K=15'+'.npz')

result_ca_K_2 = np.load('./Results/'+'CA-UCB'+'N=15'+'K=2'+'.npz')
result_ca_K_5 = np.load('./Results/'+'CA-UCB'+'N=15'+'K=5'+'.npz')
result_ca_K_10 = np.load('./Results/'+'CA-UCB'+'N=15'+'K=10'+'.npz')
result_ca_K_15 = np.load('./Results/'+'CA-UCB'+'N=15'+'K=15'+'.npz')


cu_regret_etc_K_2=result_etc_K_2['regretsucb']
cu_regret_etc_K_5=result_etc_K_5['regretsucb']
cu_regret_etc_K_10=result_etc_K_10['regretsucb']
cu_regret_etc_K_15=result_etc_K_15['regretsucb']

av_unstable_etc_K_2=result_etc_K_2['av_unstable']
av_unstable_etc_K_5=result_etc_K_5['av_unstable']
av_unstable_etc_K_10=result_etc_K_10['av_unstable']
av_unstable_etc_K_15=result_etc_K_15['av_unstable']

players_etc_K_2 = np.argmax(cu_regret_etc_K_2[:,horizon-1])
players_etc_K_5 = np.argmax(cu_regret_etc_K_5[:,horizon-1])
players_etc_K_10 = np.argmax(cu_regret_etc_K_10[:,horizon-1])
players_etc_K_15 = np.argmax(cu_regret_etc_K_15[:,horizon-1])

cu_regret_ucb_K_2=result_ucb_K_2['regretsucb']
cu_regret_ucb_K_5=result_ucb_K_5['regretsucb']
cu_regret_ucb_K_10=result_ucb_K_10['regretsucb']
cu_regret_ucb_K_15=result_ucb_K_15['regretsucb']

av_unstable_ucb_K_2=result_ucb_K_2['av_unstable']
av_unstable_ucb_K_5=result_ucb_K_5['av_unstable']
av_unstable_ucb_K_10=result_ucb_K_10['av_unstable']
av_unstable_ucb_K_15=result_ucb_K_15['av_unstable']

players_ucb_K_2 = np.argmax(cu_regret_ucb_K_2[:,horizon-1])
players_ucb_K_5 = np.argmax(cu_regret_ucb_K_5[:,horizon-1])
players_ucb_K_10 = np.argmax(cu_regret_ucb_K_10[:,horizon-1])
players_ucb_K_15 = np.argmax(cu_regret_ucb_K_15[:,horizon-1])


cu_regret_ca_K_2=result_ca_K_2['regretsucb']
cu_regret_ca_K_5=result_ca_K_5['regretsucb']
cu_regret_ca_K_10=result_ca_K_10['regretsucb']
cu_regret_ca_K_15=result_ca_K_15['regretsucb']

av_unstable_ca_K_2=result_ca_K_2['av_unstable']
av_unstable_ca_K_5=result_ca_K_5['av_unstable']
av_unstable_ca_K_10=result_ca_K_10['av_unstable']
av_unstable_ca_K_15=result_ca_K_15['av_unstable']

players_ca_K_2 = np.argmax(cu_regret_ca_K_2[:,horizon-1])
players_ca_K_5 = np.argmax(cu_regret_ca_K_5[:,horizon-1])
players_ca_K_10 = np.argmax(cu_regret_ca_K_10[:,horizon-1])
players_ca_K_15 = np.argmax(cu_regret_ca_K_15[:,horizon-1])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5,5), )

#ax1.figure(dpi = 200)

ax1.errorbar(range(horizon),cu_regret_etc_K_2[players_etc_K_2][:horizon],fmt = '-', yerr=np.std(cu_regret_etc_K_2[players_etc_K_2][:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=2', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_etc_K_5[players_etc_K_5][:horizon],fmt = '--', yerr=np.std(cu_regret_etc_K_5[players_etc_K_5][:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=5', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_etc_K_10[players_etc_K_10][:horizon],fmt = '-.', yerr=np.std(cu_regret_etc_K_10[players_etc_K_10][:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=10', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_etc_K_15[players_etc_K_15][:horizon],fmt = ':', yerr=np.std(cu_regret_etc_K_15[players_etc_K_15][:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=15', errorevery = 5000)

ax1.errorbar(range(horizon),cu_regret_ucb_K_2[players_ucb_K_2][:horizon],fmt = '-', yerr=np.std(cu_regret_ucb_K_2[players_ucb_K_2][:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=2', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ucb_K_5[players_ucb_K_5][:horizon],fmt = '--', yerr=np.std(cu_regret_ucb_K_5[players_ucb_K_5][:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=5', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ucb_K_10[players_ucb_K_10][:horizon],fmt = '-.', yerr=np.std(cu_regret_ucb_K_10[players_ucb_K_10][:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=10', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ucb_K_15[players_ucb_K_15][:horizon],fmt = ':', yerr=np.std(cu_regret_ucb_K_15[players_ucb_K_15][:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=15', errorevery = 5000)

ax1.errorbar(range(horizon),cu_regret_ca_K_2[players_ca_K_2][:horizon],fmt = '-', yerr=np.std(cu_regret_ca_K_2[players_ca_K_2][:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=2', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ca_K_5[players_ca_K_5][:horizon],fmt = '--', yerr=np.std(cu_regret_ca_K_5[players_ca_K_5][:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=5', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ca_K_10[players_ca_K_10][:horizon],fmt = '-.', yerr=np.std(cu_regret_ca_K_10[players_ca_K_10][:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=10', errorevery = 5000)
ax1.errorbar(range(horizon),cu_regret_ca_K_15[players_ca_K_15][:horizon],fmt = ':', yerr=np.std(cu_regret_ca_K_15[players_ca_K_15][:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=15', errorevery = 5000)
        

        
       
#ax1.legend()
ax1.set_xlabel('Time',fontsize = 16)
ax1.set_ylabel('Expected Cumulative Regret',fontsize = 16)
ax1.set_ylim(10, 25000)
ax1.set_yscale('log')
ax1.set_title("(c) Cumulative Regret",fontsize = 16)
#ax1.savefig('./CMP_Cumulative_Regret_varying_K.pdf')
#ax1.close()


#ax2.figure(dpi = 200)
ax2.errorbar(range(horizon),av_unstable_etc_K_2[:horizon],fmt = '-', yerr=np.std(av_unstable_etc_K_2[:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=2', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_etc_K_5[:horizon],fmt = '--', yerr=np.std(av_unstable_etc_K_5[:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=5', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_etc_K_10[:horizon],fmt = '-.', yerr=np.std(av_unstable_etc_K_10[:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=10', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_etc_K_15[:horizon],fmt = ':', yerr=np.std(av_unstable_etc_K_15[:horizon])/np.sqrt(trials),color=colorMap[0], label='Cen-ETC, K=15', errorevery = 5000)

ax2.errorbar(range(horizon),av_unstable_ucb_K_2[:horizon],fmt = '-', yerr=np.std(av_unstable_ucb_K_2[:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=2', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ucb_K_5[:horizon],fmt = '--', yerr=np.std(av_unstable_ucb_K_5[:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=5', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ucb_K_10[:horizon],fmt = '-.', yerr=np.std(av_unstable_ucb_K_10[:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=10', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ucb_K_15[:horizon],fmt = ':', yerr=np.std(av_unstable_ucb_K_15[:horizon])/np.sqrt(trials),color=colorMap[1], label='Cen-UCB, K=15', errorevery = 5000)

ax2.errorbar(range(horizon),av_unstable_ca_K_2[:horizon],fmt = '-', yerr=np.std(av_unstable_ca_K_2[:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=2', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ca_K_5[:horizon],fmt = '--', yerr=np.std(av_unstable_ca_K_5[:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=5', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ca_K_10[:horizon],fmt = '-.', yerr=np.std(av_unstable_ca_K_10[:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=10', errorevery = 5000)
ax2.errorbar(range(horizon),av_unstable_ca_K_15[:horizon],fmt = ':', yerr=np.std(av_unstable_ca_K_15[:horizon])/np.sqrt(trials),color=colorMap[2], label='MOCA-UCB, K=15', errorevery = 5000)
        

        
       
#ax2.legend()
ax2.set_xlabel('Time',fontsize = 16)
ax2.set_ylabel('Expected Averaged Unstability',fontsize = 16)
ax2.set_title("(d) Averaged Unstability",fontsize = 16)
#ax2.savefig('./CMP_Averaged_Unstability_varying_K.pdf')
#ax2.close()
fig.subplots_adjust(right=0.95)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.legend( loc='best', borderaxespad=0)
plt.savefig('./CMP_varying_K.pdf')
plt.close(fig)