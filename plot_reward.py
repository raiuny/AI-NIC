import numpy as np
import matplotlib.pyplot as plt
from env_singlelink import env_SL

episode_length = 500000
update_interval = 200
actor_lr = 5e-4
critic_lr = 5e-4
gamma = 0.95
batch_size = 64
state_length_M = 5
alpha = 0.3
learning_interval = 100

def return_throughput(rewards):
	'''
	性能指标由吞吐量表示，吞吐量表示N时隙平均每个时隙成功传输包的数量
	:param rewards:长度10000
	:return:吞吐量，长度10000
	'''
	N = int(len(rewards)/50)
	temp_sum = 0
	throughput = []
	for i in range(len(rewards)):
		if i < N:
			temp_sum += rewards[i]
			throughput.append(temp_sum / (i+1))  #长度不满N时，平均值为除以i+1,LTT
		else:
			temp_sum += rewards[i] - rewards[i-N]
			throughput.append(temp_sum / N)     #长度满N了，平均值就为总和除以N,STT
	return throughput


for idx in range(1, 2):
	reward1_l1 = np.loadtxt(f'rewards/reward1_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
							f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt')
	reward2_l1 = np.loadtxt(f'rewards/reward2_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
							f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt')
	reward3_l1 = np.loadtxt(f'rewards/reward3_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
							f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_v1.txt')

	# l1上的吞吐量
	l1_throughput1 = return_throughput(reward1_l1)   #agent1吞吐量
	l1_throughput2 = return_throughput(reward2_l1)   #agent2吞吐量
	l1_throughput3 = return_throughput(reward3_l1)   # agent1吞吐量
	# l1_throughput4 = return_throughput(reward4_l1)   # agent2吞吐量
	# mean1 = np.mean(throughput1[-1000])
	# mean2 = np.mean(throughput2[-1000])

	l1_sum_throughput = [l1_throughput1[i] + l1_throughput2[i] + l1_throughput3[i]  for i in
						 range(episode_length)]  # l1总吞吐量
	mean1 = np.mean(round(l1_sum_throughput[-1000], 2))  # 最后逼近的值
	fig = plt.figure(figsize=(14, 6))
	plt.plot(l1_sum_throughput, c='r', label='Sum')
	plt.plot(l1_throughput1, c='b', label='agent1')
	plt.plot(l1_throughput2, c='cyan', label='agent2')
	plt.plot(l1_throughput3, c='orange', label='agent3')
	# plt.plot(l1_throughput4, c='green', label='agent4')
	plt.ylim((0,1))
	plt.xlim(0, None)
	plt.xlabel("Iterations/Slots", fontsize = 14)
	plt.ylabel("Throughput", fontsize = 14)
	plt.title("link1")
	plt.text(len(l1_sum_throughput) * 0.9, 0.85, f'sum={mean1}',
			 family='Times New Roman',  # 标注文本字体
			 fontsize=18,  # 文本大小
			 fontweight='bold',  # 字体粗细
			 color='red'  # 文本颜色
			 )
	plt.legend()

	plt.show()


