import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class MEM_PPO(nn.Module):
    def __init__(self):
        super(MEM_PPO, self).__init__()
        self.data = []
        self.learning_rate = 0.0005
        self.gamma         = 0.98
        self.lmbda         = 0.95
        self.eps_clip      = 0.1
        self.K_epoch       = 3
        self.mem_th = 0.9
        self.age_noise = 2

        self.h_d=256
        self.s_d=4
        self.a_d=2
        self.memory_size=1024
        self.fc1   = nn.Linear(self.s_d,self.h_d)
        self.fc2   = nn.Linear(self.s_d*2,self.h_d)
        self.fc_pi = nn.Linear(self.h_d,2)
        self.fc_v  = nn.Linear(self.h_d,1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.mem_build()
        
    def mem_build(self):
        self.keys = Variable(F.normalize(torch.rand(
            (self.memory_size, self.s_d))-0.5, dim=1), requires_grad=False)
        self.values = Variable(F.normalize(torch.rand(
            (self.memory_size, self.s_d))-0.5, dim=1), requires_grad=False)
        self.age = torch.zeros(self.memory_size, 1).int()
    
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
        x = torch.cat((x, v_hat_top1),dim=1).reshape(x.size(0),self.s_d*2)
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
            if v_sim.item() < self.mem_th:
                age_with_noise = self.age + self.age_noise * torch.rand((self.memory_size, 1))
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

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1, flag=flag)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
