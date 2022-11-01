import torch
import numpy as np

# market ID.


# two parameters: beta -2, -1, 0,10,100,1000; size: 5, 10, 15, 20, 30
# beta=-2: global preference
# beta=-1: random preference, vary size N
# beta = 0,10,100,1000; generate preference, N=10



Beta = [-2,-1,0,10,100,1000]
Size = [5,10,15,20,30]


# generate market
# for beta in Beta:
  
#     if beta == -2:
        # num_players=5
        # num_arms = 5
        # players_ranking = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4] ]
        # arms_ranking = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4] ]
        # players_mean_value = [np.linspace(0.9, 0.1, num_arms) for j in range(num_players)]
        # players_mean = [np.zeros([num_arms]) for j in range(num_players)]

        # # change the index of players_mean
        # for p_idx in range(num_players):
        #     for arm in range(num_arms):
        #         players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

        # np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)
    

#     elif beta == -1:
#         for num in Size:
#             num_players=num
#             num_arms = num
#             players_ranking = [ np.random.permutation(num_arms).tolist() for j in range(num_players)]
#             arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]
#             players_mean_value = [np.linspace(0.9, 0.1, num_arms) for j in range(num_players)]
#             players_mean = [np.zeros([num_arms]) for j in range(num_players)]

#             # change the index of players_mean
#             for p_idx in range(num_players):
#                 for arm in range(num_arms):
#                     players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

#             np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)
    
#     else:
#         num_players = 10
#         num_arms = 10
#         arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]
#         players_mean = [np.zeros([num_arms]) for j in range(num_players)]
    
#         # using beta to calculate the mean and then calculate the ranking
#         x = np.random.uniform(low=0.0, high=1.0, size=num_arms)
#         varepsilon = np.random.logistic(0, 1, size = (num_players,num_arms))
    
#         barmu = np.zeros([num_players,num_arms])
#         for i in range(num_players):
#             for j in range(num_arms):
#                 barmu[i,j] = beta*x[j]+varepsilon[i][j]

#         for i in range(num_players):
#             for j in range(num_arms):
#                 for arm in range(num_arms):
#                     if barmu[i][arm] <= barmu[i][j]:
#                         players_mean[i][j]+=1
             
#         players_mean = np.array(players_mean)/num_arms
 
#         players_mean = torch.from_numpy(players_mean)
#         _,ranking = players_mean.topk(num_arms, 1)
#         players_ranking = ranking.numpy().tolist()
#         np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)



# # read market
# for beta in Beta:
#     if beta==-2:
#         num_players = 5
#         market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
#         print("beta = ",beta, "num_player = ", num_players )
#         print("Player_rank", market['player_rank'])
#         print("Arm_rank", market['arm_rank'])
#         print("Player_mean", market['player_mean'])
#     elif beta==-1:
#         for num in Size:
#             num_players=num
#             print("beta = ",beta, "num_player = ", num_players )
#             market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
#             # print("Size", len(market['player_rank']))
#             print("Player_rank", market['player_rank'])
#             print("Arm_rank", market['arm_rank'])
#             print("Player_mean", market['player_mean'])
#     else:
#         num_players = 10
#         print("beta = ",beta, "num_player = ", num_players )
#         market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
#         # print("Size", len(market['player_rank']))
#         print("Player_rank", market['player_rank'])
#         print("Arm_rank", market['arm_rank'])
#         print("Player_mean", market['player_mean'])




# vary the Delta, N=5

# start = [0.1, 0.2, 0.3, 0.4]
# end =   [0.9, 0.8, 0.7, 0.6]

# num_players=5
# num_arms = 5


# players_ranking = [ np.random.permutation(num_arms).tolist() for j in range(num_players)]
# arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]


# for s in range(len(start)):
#     players_mean_value = [np.linspace(end[s],start[s], num_arms) for j in range(num_players)]
#     players_mean = [np.zeros([num_arms]) for j in range(num_players)]

#     # change the index of players_mean
#     for p_idx in range(num_players):
#         for arm in range(num_arms):
#             players_mean[p_idx][arm] = players_mean_value[p_idx][players_ranking[p_idx].index(arm)]

#     np.savez('./Markets/beta_'+str(start[s])+'N_'+str(num_players)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)


    

# num_players=5
# for s in range(len(start)):
#     beta = start[s]
#     print("beta = ",beta, "num_player = ", num_players )
#     market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'.npz')
#     # print("Size", len(market['player_rank']))
#     print("Player_rank", market['player_rank'])
#     print("Arm_rank", market['arm_rank'])
#     print("Player_mean", market['player_mean'])




# vary beta, N=5
num_players = 10
num_arms = 5
Beta = [0,10,50, 100]
arms_ranking = [ np.random.permutation(num_players).tolist() for j in range(num_arms)]

for beta in Beta:
        players_mean = [np.zeros([num_arms]) for j in range(num_players)]
    
        # using beta to calculate the mean and then calculate the ranking
        x = np.random.uniform(low=0.0, high=1.0, size=num_arms)
        varepsilon = np.random.logistic(0, 1, size = (num_players,num_arms))
    
        barmu = np.zeros([num_players,num_arms])
        for i in range(num_players):
            for j in range(num_arms):
                barmu[i,j] = beta*x[j]+varepsilon[i][j]

        for i in range(num_players):
            for j in range(num_arms):
                for arm in range(num_arms):
                    if barmu[i][arm] <= barmu[i][j]:
                        players_mean[i][j]+=1
             
        players_mean = np.array(players_mean)/num_arms
 
        players_mean = torch.from_numpy(players_mean)
        _,ranking = players_mean.topk(num_arms, 1)
        players_ranking = ranking.numpy().tolist()
        np.savez('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'K_'+str(num_arms)+'.npz', player_rank = players_ranking, arm_rank=arms_ranking ,player_mean = players_mean)


# read

for beta in Beta:
        print("beta = ",beta, "num_player = ", num_players )
        market = np.load('./Markets/beta_'+str(beta)+'N_'+str(num_players)+'K_'+str(num_arms)+'.npz')
        # print("Size", len(market['player_rank']))
        print("Player_rank", market['player_rank'])
        print("Arm_rank", market['arm_rank'])
        print("Player_mean", market['player_mean'])


