import os 


path = './plots/DuelingDDQNAgent_BoxingNoFrameskip-v4_lr0.0001_1000games.txt'
alg = 'DuelingDDQNAgent'
game = "boxing"
with open(path, 'r') as f:
    data = f.readlines()
f.close()


graph_data = {"Episode":[], "Epsilon":[], "Reward":[], 'Last_100_Avg_Rew':[], 'Avg_Rew': [], 'Step':[]}
for d in data:
    line = d.split()
    graph_data['Episode'].append(int(line[0].split(':')[1]))
    graph_data['Epsilon'].append(float(line[1].split(':')[1]))
    graph_data['Reward'].append(float(line[2].split(':')[1]))
    graph_data['Last_100_Avg_Rew'].append(float(line[3].split(':')[1]))
    graph_data['Avg_Rew'].append(float(line[4].split(':')[1]))
    graph_data['Step'].append(float(line[5].split(':')[1]))


def write_file_reward(filename_, graph_data, n=None):
    with open(filename_, 'w') as f:
        f.write('x' + '\t' + 'fx' + '\n')
        if n==None:
            for ep, d in zip(graph_data['Episode'], graph_data['Reward']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
        else:
            i = 0
            for ep, d in zip(graph_data['Episode'], graph_data['Reward']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
                i+=1 
                if i>=n: break 
    f.close()
    
    
def write_file_avg_last_reward(filename_, graph_data, n = None):
    with open(filename_, 'w') as f:
        f.write('x' + '\t' + 'fx' + '\n')
        if n==None:
            for ep, d in zip(graph_data['Episode'], graph_data['Last_100_Avg_Rew']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
        else:
            i = 0
            for ep, d in zip(graph_data['Episode'], graph_data['Last_100_Avg_Rew']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
                i+=1 
                if i>=n: break 
    f.close()

def write_file_avg_reward(filename_, graph_data, n = None):
    with open(filename_, 'w') as f:
        f.write('x' + '\t' + 'fx' + '\n')
        if n==None:
            for ep, d in zip(graph_data['Episode'], graph_data['Last_100_Avg_Rew']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
        else:
            i = 0
            for ep, d in zip(graph_data['Episode'], graph_data['Avg_Rew']): 
                f.write(str(ep) + '\t' + str(d) + '\n')
                i+=1 
                if i>=n: break 
    f.close()
        
  
def write_file_epsilon(filename_, graph_data):
    with open(filename_, 'w') as f:
        f.write('x' + '\t' + 'fx' + '\n')
        for ep, d in zip(graph_data['Episode'], graph_data['Epsilon']): 
            f.write(str(ep) + '\t' + str(d) + '\n')
    f.close()

###################################################################################
write_file_reward(alg + '_' + game +'_reward.dat', graph_data)
write_file_avg_last_reward(alg + '_' + game +'_last100avg_reward.dat', graph_data)
write_file_avg_reward(alg + '_' + game +'_avg_reward.dat', graph_data)

#write_file_epsilon(alg+'epsilon.dat', graph_data)
