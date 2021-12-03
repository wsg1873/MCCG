import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
from rlcore.distributions import DiagGaussian

import torch.nn.functional as F
import math
import random
import time
from torch.distributions import Normal
# from GAT import GraphAttentionLayer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


# input_size:为观测向量维度 10，
class MPNN(nn.Module):
    def __init__(self, args_value, action_space, num_other_agents, num_agents, num_entities, role, input_size=16, hidden_dim=128,
                 embed_dim=None,
                 pos_index=0, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, entity_mp=False):
        super().__init__()
        self.env_name = args_value.env_name
        self.args = args_value
        self.obs_masking_dist = args_value.obs_masking_dist
        self.role = role
        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.num_agents = num_agents  # number of agents
        self.our_group = num_agents
        self.other_group = num_other_agents
        self.num_entities = num_entities  # number of entities
        self.K = 3  # message passing rounds
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size  #
        self.entity_mp = entity_mp  # False
        # this index must be from the beginning of observation vector
        self.pos_index = pos_index  # ？
        self.K_nearest = args_value.k_nearest
        self.dt = torch.tensor(0.1).to(self.args.device)

        self.w_0_our_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))

        self.w_0_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_1_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_2_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_3_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_4_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))

        self.w_our_spread = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))

        self.w_entity_spread = nn.Sequential(nn.Linear(2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_entity1_spread = nn.Sequential(nn.Linear(2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_entity2_spread = nn.Sequential(nn.Linear(2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_entity3_spread = nn.Sequential(nn.Linear(2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.w_entity4_spread = nn.Sequential(nn.Linear(2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.entity_update_our_cat = nn.Sequential(nn.Linear(self.h_dim*2, self.h_dim),
                                     self.nonlin(inplace=True))
        self.gat_our_st = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_st = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_st1 = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_st2 = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_st3 = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_st4 = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)
        self.gat_entity_stsum = GraphAttentionLayer(args = self.args,in_features=self.h_dim, out_features=self.h_dim, dropout=0, alpha=0.2,
                                                 concat=True).to(self.args.device)





        self.messages = MultiHeadAttention(n_heads=self.n_heads, input_dim=self.h_dim, embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, self.h_dim),
                                    self.nonlin(inplace=True))

        self.obs_messages = MultiHeadAttention(n_heads=self.n_heads, input_dim=self.h_dim, embed_dim=self.embed_dim)

        self.obs_update = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, self.h_dim),
                                    self.nonlin(inplace=True))

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim, 1))

        self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True))


        # self.mu_head = nn.Sequential(nn.Linear(self.h_dim, 2), nn.Softplus())
        # self.sigma_head = nn.Sequential(nn.Linear(self.h_dim, 2), nn.Softplus())

        # if self.entity_mp:
        self.encoder = nn.Sequential(nn.Linear(4, self.h_dim),
                                     self.nonlin(inplace=True))
        self.entity_encoder = nn.Sequential(nn.Linear(4, self.h_dim),
                                            self.nonlin(inplace=True))

        self.entity_messages = MultiHeadAttention(n_heads=1, input_dim=self.h_dim, embed_dim=self.embed_dim)

        self.entity_update = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, self.h_dim),
                                           self.nonlin(inplace=True))


        self.hard_bi_GRU = nn.GRU(self.h_dim * 2, self.h_dim, bidirectional=True)
        self.hard_encoding = nn.Linear(self.h_dim * 2, 2)


        # num_actions = action_space.n  # 5 Discrete(5)
        num_actions = action_space.shape[0]
        # num_actions = 2

        self.dist = Categorical(self.h_dim, num_actions)
        self.dist_continous = DiagGaussian(self.h_dim, num_actions)

        self.alpha_head = nn.Sequential(nn.Linear(self.h_dim, num_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(self.h_dim, num_actions), nn.Softplus())
        
        self.is_recurrent = False

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))
        self.com_attn_mat = np.zeros((self.K, num_agents, num_agents))

        self.our_attn_mat = np.zeros((num_agents, num_agents))
        self.target_attn_mat = np.zeros((num_agents, self.num_entities))

        self.dropout_mask = None

    def calculate_mask(self, inp):
        # inp is batch_size x self.input_size where batch_size is num_processes*num_agents

        pos = inp[:, self.pos_index:self.pos_index + 2]  #
        bsz = inp.size(0) // self.num_agents  #
        mask = torch.full(size=(bsz, self.num_agents, self.num_agents), fill_value=0, dtype=torch.bool)  #

        if self.mask_dist is not None and self.mask_dist > 0:
            for i in range(1, self.num_agents):
                shifted = torch.roll(pos, -bsz * i,
                                     0)  # torch.roll(input,shits,dim)
                dists = torch.norm(pos - shifted, dim=1)  # dim=x，
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:, x, (x + i) % self.num_agents].copy_(restrict[bsz * x:bsz * (x + 1)])  #
        # dropout_masking ，mask_dist=-10
        elif self.mask_dist is not None and self.mask_dist == -10:
            if self.dropout_mask is None or bsz != self.dropout_mask.shape[0] or np.random.random_sample() < 0.1:  # sample new dropout mask
                temp = torch.rand(mask.size()) > 0.85
                temp.diagonal(dim1=1, dim2=2).fill_(0)
                self.dropout_mask = (temp + temp.transpose(1, 2)) != 0
            mask.copy_(self.dropout_mask)

        return mask



    def _fwd_spread_54(self, inp, inp_1, inp_2, inp_3, inp_4):
        # print(type(inp))
        agent_inp = inp[:, :self.input_size]  #
        bsz = agent_inp.size(0)#
        mask = self.calculate_mask(agent_inp)  # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted

        #
        h_our,h_our_atten = self.obs_gat_our(inp) # h(bsz,  self.dim), inp (bsz,input_size）
        h_target,h_target_atten = self.obs_gat_target(inp) # h(bsz,  self.dim), inp（bsz,input_size）
        # print('h_our_atten',h_our_atten[:,0,:], h_our_atten[:,0,:].sum(1))
        # print('h_target_atten',h_target_atten[:,0,:],h_target_atten[:,0,:].sum(1))
        self.our_attn_mat=h_our_atten[:,0,:]
        self.target_attn_mat=h_target_atten[:,0,:]

        h = self.entity_update_our_cat(torch.cat((h_our, h_target), 1))  # should be (batch_size,self.h_dim),


        h = h.view(self.num_agents, -1, self.h_dim).transpose(0, 1)  # should be (batch_size/N,N,self.h_dim)#?32x3x128

        for k in range(self.K):
            m, attn = self.messages(h, mask=mask, return_attn=True)  # should be <batch_size/N,N,self.embed_dim>
            h = self.update(torch.cat((h, m), 2))  # should be <batch_size/N,N,self.h_dim>
        h = h.transpose(0, 1).contiguous().view(-1, self.h_dim)  #

        self.attn_mat = attn.squeeze().detach().cpu().numpy()
        # print('attn_mat',self.attn_mat)
        return h

    def obs_gat_our(self,inp):
        agent_inp = inp[:, :self.input_size]  #
        bsz = agent_inp.size(0)#
        # mask = self.calculate_mask(agent_inp)  # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted
        agent_state = agent_inp[:, self.pos_index:self.pos_index + 4]  #
        # agent_pos = agent_inp[:, self.pos_index:self.pos_index + 2]  #
        agent_h0 = self.w_0_our_spread(agent_state)

        entity_inp = inp[:, 4:4*self.our_group]  #
        entity_inp = entity_inp.view(bsz, self.our_group-1, 4)  # agent_inp:(bsz,K_nearest, 4)
        entity_inp = self.w_our_spread(entity_inp)
        entity_inp = entity_inp.view(bsz, -1)  # agent_inp:(bsz,K_nearest, 4)
        entity_inp = torch.cat((agent_h0, entity_inp), 1)
        # entity_mask = self.observation_entity_mask(agent_inp)
        entity_mask = torch.full(size=(bsz, self.our_group, self.our_group), fill_value=0,
                          dtype=torch.bool).to(self.args.device)  #

        entity_mask[:, 0, 0] = 0
        agent_pos=inp[:,:4*self.our_group].view(bsz, self.our_group,4).detach()
        agent_pos=agent_pos[:,:,self.pos_index:self.pos_index+2] #B,N,2
        data=inp[:, :2].repeat(1, self.our_group ).view(bsz,self.our_group,2).detach()
        data[:,0,:]=0
        agent_pos =agent_pos+data
        # print('agent_pos',agent_pos)
        # print('agent_pos',agent_pos)
        pi=agent_pos.repeat(1,self.our_group,1).view(bsz,self.our_group,self.our_group,2)
        pj=agent_pos.repeat(1,self.our_group,1).view(bsz,self.our_group,self.our_group,2).transpose(1, 2)
        dist=torch.norm((pi-pj),dim=3)<=self.mask_dist
        entity_mask=~dist.detach()


        entity_inp = entity_inp.view(bsz, self.our_group, self.h_dim)  # agent_inp:(bsz,K_nearest, 4)
        entity_m, entity_atten = self.gat_our_st(entity_inp, entity_mask)  # m(bsz,self.K_nearest,c)
        entity_h = entity_m[:, 0, :]
        return entity_h,entity_atten

    def obs_gat_target(self, inp):
        agent_inp = inp[:, :self.input_size]  #
        bsz = agent_inp.size(0)  #
        # mask = self.calculate_mask(agent_inp)  # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted
        agent_state = agent_inp[:, self.pos_index:self.pos_index + 4]  #
        # agent_pos = agent_inp[:, self.pos_index:self.pos_index + 2]  #
        agent_h0 = self.w_0_spread(agent_state)

        entity_inp = inp[:, 4*self.our_group:]  #
        entity_inp = entity_inp.view(bsz, self.num_entities, 2)  # agent_inp:(bsz,K_nearest, 4)
        entity_inp = self.w_entity_spread(entity_inp)
        entity_inp = entity_inp.view(bsz, -1)  # agent_inp:(bsz,K_nearest, 4)
        entity_inp = torch.cat((agent_h0, entity_inp), 1)
        # entity_mask = self.observation_entity_mask(agent_inp)
        entity_mask = torch.full(size=(bsz, self.num_entities + 1, self.num_entities + 1), fill_value=0,
                                 dtype=torch.bool).to(self.args.device)  #
        # entity_mask[:, 0, 0] = 0
        entity_inp = entity_inp.view(bsz, self.num_entities + 1, self.h_dim)  # agent_inp:(bsz,K_nearest, 4)
        entity_m, entity_atten = self.gat_entity_st(entity_inp, entity_mask)  # m(bsz,self.K_nearest,c)
        entity_h = entity_m[:, 0, :]
        return entity_h,entity_atten


    def _fwd(self,inp,inp_1, inp_2 , inp_3, inp_4):
        if self.env_name == "simple_spread":
            h = self._fwd_spread_54(inp,inp_1, inp_2, inp_3, inp_4)

        else:
            print('False')

        return h


    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def _value(self, x):
        return self.value_head(x)

    def _policy(self, x):
        return self.policy_head(x)

    def actionsconvert(self, action):
        action_u = torch.zeros(2)
        if action == 1: action_u[0] = -1.0
        if action == 2: action_u[0] = +1.0
        if action == 3: action_u[1] = -1.0
        if action == 4: action_u[1] = +1.0
        return action_u*0.5

    def matrixPow(self, Matrix, n):
        if (type(Matrix) == list):
            Matrix = np.array(Matrix)
        if (n == 1):
            return Matrix
        else:
            return np.matmul(Matrix, self.matrixPow(Matrix, n - 1))



    ##########################no fitering#####################################
    def act_nofiltering(self, inp, inp_1, inp_2, inp_3, inp_4, state, mask=None, deterministic=False):
        x = self._fwd(inp,inp_1, inp_2, inp_3, inp_4)
        value = self._value(x)
        # dist = self.dist(self._policy(x))
        alpha = self.alpha_head(self._policy(x)) + 1
        beta = self.beta_head(self._policy(x)) + 1
        # dist = self.dist_continous(self._policy(x))  # dist:
        policy_dist = torch.distributions.Beta(alpha, beta)
        # action = policy_dist.sample()
        if deterministic:  # False
            action = alpha / (alpha + beta)

        else:
            action = policy_dist.sample()
        # print(action)
        # print(action.size())
        # print('111')
        # print(policy_dist.log_prob(action).size())
        log_prob = policy_dist.log_prob(action).sum(-1).unsqueeze(-1)

        return value,action,log_prob,state



    ##########################fitering#####################################
    ##########################fitering#####################################
    def act(self, inp, inp_1, inp_2, inp_3, inp_4, state, mask=None, deterministic=False):

        if self.args.train:
            value, action, log_prob, state = self.act_nofiltering(inp, inp_1, inp_2, inp_3, inp_4, state, mask, deterministic)
        else:
            value, action, log_prob, state = self.act_filtering(inp, inp_1, inp_2, inp_3, inp_4, state, mask, deterministic)
        return value, action, log_prob, state

    def act_filtering(self, inp, inp_1, inp_2, inp_3, inp_4, state, mask=None, deterministic=False):

        x = self._fwd(inp,inp_1, inp_2, inp_3, inp_4)
        value = self._value(x)
        # dist = self.dist(self._policy(x))
        alpha = self.alpha_head(self._policy(x)) + 1
        beta = self.beta_head(self._policy(x)) + 1
        # dist = self.dist_continous(self._policy(x))  # dist:
        policy_dist = torch.distributions.Beta(alpha, beta)

        # print('alpha, beta',alpha, beta)

        # action = policy_dist.sample()
        if deterministic:  # False
            action = alpha / (alpha + beta)

        else:
            action = policy_dist.sample()
        # print('action',action)

        state_batch = inp[:,:4]
        state_batch = state_batch.view(self.num_agents, -1, 4).transpose(0, 1)  # should be (batch_size/N,N,self.h_dim)#?32x3x128
        pos_batch = state_batch[:,:,:2] #32x3x2
        vel_batch = state_batch[:,:,2:4] #32x3x2
        bsz = inp.size(0)//self.num_agents
        action_list = action.view(self.num_agents,-1,2).transpose(0, 1)
        for k in range(bsz):
            pi = pos_batch[k].repeat(self.num_agents, 1).view(self.num_agents, self.num_agents, 2).transpose(0, 1)
            pj = pos_batch[k].repeat(self.num_agents, 1).view(self.num_agents, self.num_agents, 2)
            pij = pi - pj

            # print('pij',torch.norm((pi - pj), dim=2) )
            connections_high = torch.norm((pi - pj), dim=2) <= self.mask_dist #
            connections_low = torch.norm((pi - pj), dim=2) <= (self.mask_dist - 0.1)
            critical_matrix_origin = connections_high.float() - connections_low.float()#

            adj_matrix = connections_high.float() - torch.eye(self.num_agents).to(self.args.device) #
            for i in range(self.num_agents):
                if critical_matrix_origin[i].sum(0)>0: #
                    for j in range(self.num_agents):#
                        if j==i: continue
                        if critical_matrix_origin[i,j] > 0 and (connections_low[i,:]*connections_low[j,:]).sum(0)>0: #
                            critical_matrix_origin[i,j] = 0
                            critical_matrix_origin[j,i] = 0
                            # temp= (connections_low[i,:]*connections_low[j,:]).sum(0)>0
            if critical_matrix_origin.sum()>0:

                N_ig_matrix = torch.zeros(self.num_agents,self.num_agents) #
                # N_ig_matrix = np.zeros([self.num_agents,self.num_agents]) #
                # print('distance',(torch.norm(pij, dim=2)))
                # print(critical_matrix_origin)

                I= torch.eye(self.num_agents).to(self.args.device)
                # temp_adj_matrix = connections_high.float() - torch.eye(self.num_agents).to(self.args.device) #
                # print(temp_adj_matrix)
                for i in range(self.num_agents):#
                    if critical_matrix_origin[i].sum() > 0 and adj_matrix[i].sum()>1: #
                        for j in range(self.num_agents):#
                            temp_adj_matrix = connections_high.float() - torch.eye(self.num_agents).to(self.args.device) #
                            # print('temp_adj_matrix',temp_adj_matrix)

                            if j==i: continue
                            if temp_adj_matrix[i,j] == 1:
                                temp_adj_matrix[i,j] = 0
                                temp_adj_matrix[j,i] = 0
                                # print('temp_adj_matrix',temp_adj_matrix)

                                reach_matrix = self.matrixPow((temp_adj_matrix + I).data.cpu().numpy(), self.num_agents) #
                                # print('reach_matrix',reach_matrix)
                                # print(i,j)
                                if reach_matrix [i,j]> 0: #
                                    N_ig_matrix[i,j] = 1
                                    N_ig_matrix[j,i] = 1
                if N_ig_matrix.sum() > 0: #
                    #
                    u = action_list[k].repeat(self.num_agents, 1).view(self.num_agents, self.num_agents, 2).transpose(0, 1)
                    #
                    cos_theta = (-pij[:,:,0]*u[:,:,0] + -pij[:,:,1]*u[:,:,1])/((torch.norm(pij, dim=2)+1e-6)*torch.norm(u, dim=2)+1e-6) #
                    theta = cos_theta/math.pi*180
                    N_ig_matrix = N_ig_matrix.to(self.args.device)
                    # print(cos_theta,N_ig_matrix,1-N_ig_matrix)
                    # print(theta*N_ig_matrix + (1-N_ig_matrix)*(1e6*torch.ones(self.num_agents).to(self.args.device)),1 )
                    [values,indices] = torch.min(theta*N_ig_matrix + (1-N_ig_matrix)*(1e6*torch.ones(self.num_agents).to(self.args.device)),1 ) #
                    # N_ig_matrix()
                    # print('')
                    # print(N_ig_matrix)

                    for i in range(self.num_agents):
                        N_ig_matrix[i, indices[i]] = 0 #
                    # print(N_ig_matrix)
                    remove_matrix = N_ig_matrix.transpose(0,1)*N_ig_matrix #
                    # print(remove_matrix)
                    remove_matrix = critical_matrix_origin * remove_matrix

                    critical_matrix_final = critical_matrix_origin - remove_matrix #
                    # print('')
                    # print(critical_matrix_final)
                else:
                    critical_matrix_final = critical_matrix_origin
                #
                for i in range(self.num_agents):
                    if critical_matrix_final[i,:].sum() > 0:
                        # print('123')
                        # print(critical_matrix_final)
                        # print((self.mask_dist - torch.norm((pi - pj), dim=2)))
                        # a+1e6*(critical_matrix_final == 0).float()
                        # r-dij
                        aa = ((self.mask_dist - torch.norm((pi - pj), dim=2)) * torch.tensor(critical_matrix_final))
                        # print(aa)
                        # print(aa+1e6*(critical_matrix_final == 0).float())

                        # print(((self.mask_dist - torch.norm((pi - pj), dim=2))*torch.tensor(critical_matrix_final)))
                        # print(torch.min((aa + 1e6*(critical_matrix_final == 0).float())[i])/2.0)
                        u_max = min(torch.min((aa + 1e6*(critical_matrix_final == 0).float())[i]) / 2.0,0.05)
                        if u_max <=1e-4:
                            u_max = 0
                        # u_max = ((torch.min(((self.mask_dist - torch.norm((pi - pj), dim=2))*torch.tensor(critical_matrix_final))[i]+bb))/2.0).float()
                        # print('max moving length')
                        # print(i,u_max)
                        # print(action_list[k,i])
                        # print(torch.norm(action_list[k,i],dim=0))
                        # print('1111')
                        # print(torch.norm(action_list[k, i]))
                        u_y = action_list[k, i]*torch.tensor([10.0,10.0]).to(self.args.device)+torch.tensor([-5.0,-5.0]).to(self.args.device)
                        # print(i,u_y)
                        u_y_len=torch.norm(u_y, dim=0)*0.01
                        # print('u_y_len',u_y_len)
                        # print('u_max',u_max)

                        # print()
                        # print('action_list_old',action_list[k,i])

                        if u_y_len >= u_max:
                            u_limit = u_y / (torch.norm(u_y, dim=0)+1e-10) *u_max*100 #-5，5
                        # (u_limit + 5) / 10
                        # print(action_list[k, i]/torch.norm(action_list[k,i],dim=0))
                        # action_i = action_list[k,i]/torch.norm(action_list[k,i],dim=2)*u_max
                            action_list[k,i] = (u_limit + 5) / 10

                        # print('action_list_new',action_list[k,i])
        action_list = action_list.transpose(0, 1).contiguous().view(-1,2)
        # print(action_list)

        log_prob = policy_dist.log_prob(action_list).sum(-1).unsqueeze(-1)

        return value,action_list,log_prob,state



    def evaluate_actions(self, inp, inp_1, inp_2, inp_3, inp_4,state, mask, action):
        x = self._fwd(inp, inp_1, inp_2, inp_3, inp_4)
        value = self._value(x)
        # dist = self.dist(self._policy(x))
        # dist = self.dist_continous(self._policy(x))  # dist
        alpha = self.alpha_head(self._policy(x)) + 1
        beta =  self.beta_head(self._policy(x)) + 1
        # dist = self.dist_continous(self._policy(x))  # dist
        policy_dist = torch.distributions.Beta(alpha, beta)

        log_prob = policy_dist.log_prob(action).sum(-1).unsqueeze(-1)

        entropy = policy_dist.entropy().mean()


        return value,log_prob,entropy,state



    def get_value(self, inp, inp_1, inp_2, inp_3, inp_4,state, mask):
        x = self._fwd(inp, inp_1, inp_2, inp_3, inp_4)
        value = self._value(x)
        return value


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, args,in_features, out_features, dropout, alpha, concat=True):
        """
        Dense version of GAT.
        :param in_features:
        :param out_features:
        :param dropout: dropout
         param alpha Relu0.2
        :param alpha: LeakyRelu
        """

        super(GraphAttentionLayer, self).__init__()
        self.args = args
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # print(in_features)
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        # self.W = nn.Parameter(torch.zeros(128, 32))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print(B,N,C)
        h = torch.matmul(input, self.W)
        B, N, C = h.size()  # B N out_features
        # (B,N,C) (B,N,N*C),(B,N*N,C),(B,N*N,2C)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                                  2 * self.out_features)  # [B,N,N,2C]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (B,N,N),
        #print(type(e))
        # print(self.a.shape)

        zero_vec = -9e15 * torch.ones_like(e).to(self.args.device)
        # （condition, the value in the condition, 0）
        attention = torch.where(adj < 1, e, zero_vec)  #

        attention = F.softmax(attention, dim=2)  # B,N,N

        h_prime = torch.matmul(attention, h)  # (B N N B N C)=B N C

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        if return_attn:
            return out, attn
        return out