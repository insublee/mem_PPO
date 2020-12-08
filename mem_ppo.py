import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
mem_th = 0.9
age_noise = 2

h_d=256
s_d=4
a_d=2
memory_size=1024

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(s_d,h_d)
        self.fc2   = nn.Linear(s_d*2,h_d)
        self.fc_pi = nn.Linear(h_d,2)
        self.fc_v  = nn.Linear(h_d,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.mem_build()
        
    def mem_build(self):
        self.keys = Variable(F.normalize(torch.rand(
            (memory_size, s_d))-0.5, dim=1), requires_grad=False)
        self.values = Variable(F.normalize(torch.rand(
            (memory_size, s_d))-0.5, dim=1), requires_grad=False)
        self.age = torch.zeros(memory_size, 1).int()
    
    def retrive(self,x,topk):
        k_sim = torch.matmul(x, self.keys.T)
        k_sim, k_idx = torch.topk(k_sim, topk, dim=1)
        v_hat = self.values[k_idx.squeeze(1)]
        
        return v_hat, k_sim, k_idx

        
    def pi(self, x, softmax_dim = 1, flag=False):
        # 연관기억 뽑고 v_hat[batch, topk, s_d]
        v_hat, k_sim, k_idx = self.retrive(x,topk=4)
        if flag:
            print("v_hat :", v_hat.size())
        
        # 가치있는것을 고르고 predicted_values[batch, topk, 1]
        predicted_values = self.v(v_hat)
        if flag:
            print("predicted_values :", predicted_values.size())
        
        # idx[batch, 1]
        idx = torch.argsort(predicted_values, dim=1)
        idx = idx.expand_as(v_hat)
        #print(v_hat[idx==0].reshape(-1, 4))
        v_hat_top1 = v_hat[idx==0].reshape(x.size())
        if flag:
            print("x :", x.size())
            print("v_hat_top1 :", v_hat_top1.size())
        
        # 그 정보도 같이 넣어준다. x[batch,s_d]
        x = torch.cat((x, v_hat_top1),dim=1).reshape(x.size(0),s_d*2)
        #x = torch.cat((x, v_hat_top1),dim=1)
        if flag:
            #print(" torch.cat((x, v_hat_top1),dim=1).squeeze(0) : ",x.size())
            print("torch.cat((x, v_hat_top1),dim=1) : ",x.size())
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        if flag:
            print("self.fc_pi(x) :", x.size())
        prob = F.softmax(x, dim=softmax_dim)
        if flag:
            print("prob :", prob, prob.size())
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        s, a, r, s_prime, prob_a, done = transition
        
        with torch.no_grad():
            s_t = F.normalize(torch.from_numpy(s).float().unsqueeze(0),dim=1)
            s_p_t = F.normalize(torch.from_numpy(s_prime).float().unsqueeze(0),dim=1)
            
            v_hat, k_sim, k_idx = self.retrive(s_t, topk=1)
            v_sim = torch.mm(v_hat, s_p_t.T)
            
            # 예측이랑 실제랑 차이가 많이 남
            if v_sim.item() < mem_th:
                age_with_noise = self.age + age_noise * torch.rand((memory_size, 1))
                oldest_indices = torch.argmax(age_with_noise)

                self.keys[oldest_indices] = s_t
                self.values[oldest_indices] = s_p_t
                self.age[oldest_indices] = 0

            # 예측이랑 실제랑 차이가 별로 안남
            else:
                self.age[k_idx] = 0 
        
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self, flag=False):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1, flag=flag)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
