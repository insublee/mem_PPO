total_lst = []
print_interval = 10
episodes = 1000
tryes = 100
    
for i in range(tryes):
    print(i,"th trys")
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    
    
    try_lst = []
    for n_epi in range(1,episodes+1):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                s_t = F.normalize(torch.from_numpy(s).float().unsqueeze(0),dim=1)
                prob = model.pi(s_t).squeeze()
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s_p_t = F.normalize(torch.from_numpy(s_prime).float().unsqueeze(0),dim=1)

                s = s_prime

                score += r
                if done:
                    break

            model.train_net(flag=False)

        if n_epi%print_interval==0 and n_epi!=0:
            #print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            try_lst.append(score/print_interval)
            score = 0.0
    total_lst.append(try_lst)
    env.close()
arr = np.array(total_lst)
episode_mean = np.mean(arr, axis=0)
print("10에피소드당 평균 : ",episode_mean)
episode_std = np.std(arr, axis=0)
print("10에피소드당 분산 : ",episode_std)
plt.xticks(list(range(int(episodes/print_interval))),list(range(10,episodes+1,10)))
plt.plot(episode_mean, label="episode_mean")
plt.plot(episode_std, label="episode_std")
