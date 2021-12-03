import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class Scenario(BaseScenario):
    def __init__(self, num_good_agents=3,  num_adversaries=3,  dist_threshold=0.5, arena_size=1, identity_size=0):
        self.num_agents = num_good_agents
        # print(self.num_agents)
        self.landmarks = num_adversaries
        self.rewards = np.zeros(self.num_agents)
        self.temp_done = False
        self.dist_threshold = dist_threshold #coverage range
        self.arena_size = arena_size
        self.identity_size = identity_size
        self.connection_distres = 1.0 #communication range
        self.unconnections = 0
        self.coverage_rate_sum = 0

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = self.num_agents
        num_landmarks = self.landmarks
        world.collaborative = False
        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
            agent.adversary = False
            agent.learning = True

        # add landmarks
        world.landmarks = [Landmark(iden = i) for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1

            landmark.max_speed = 0.2 if landmark.movable else 0
            landmark.velocity_callback = self.velocity_callback
            landmark.learning = False

        # make initial conditions
        self.reset_world(world)
        world.dists = []
        world.dist_thres = self.dist_threshold
        return world
    def velocity_callback(self,landmark, world):
        # print(np.random(world.dim_p))
        # landmark.state.p_vel = (2*np.random.rand(world.dim_p) - np.ones(world.dim_p))*0.1
        # a = 0

        I = landmark.state.p_vel

        if world.steps % 20 ==0 :
            I = (2*np.random.rand(world.dim_p) - np.ones(world.dim_p))* landmark.max_speed
            # I = np.array(target_vel[landmark.iden])* landmark.max_speed

        N = np.zeros(world.dim_p)
        if abs(landmark.state.p_pos[0]) > self.arena_size*1.0:
            N[0] = landmark.state.p_pos[0]/abs(landmark.state.p_pos[0])
        if abs(landmark.state.p_pos[1]) > self.arena_size*1.0:
            N[1] = landmark.state.p_pos[1]/abs(landmark.state.p_pos[1])
        velocity = I - 2.0*np.dot(N,I)*N
        # if
        return velocity*10
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            color_matrix = 0.5*np.eye(self.num_agents)
            # agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = color_matrix[i]


        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])


        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-self.connection_distres/1.4/2, self.connection_distres/1.4/2, world.dim_p)
            # agent.state.p_pos = np.array([-self.arena_size + self.arena_size * 2 / 5 * (i + 0.5), -0.5])
            # agent.state.p_pos = np.array(agent_pos[i])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            # print(agent.state.p_pos)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.array(target_pos[i])
            # landmark.state.p_vel = np.array(target_vel[i]) * landmark.max_speed

            landmark.state.p_pos = np.random.uniform(-self.arena_size*1.0, self.arena_size*1.0, world.dim_p)
            landmark.state.p_vel = (2*np.random.rand(world.dim_p) - np.ones(world.dim_p))* landmark.max_speed

        # landmark_reach = True
        # while(landmark_reach):
        #     for i, landmark in enumerate(world.landmarks):
        #         landmark.state.p_pos = np.random.uniform(-self.arena_size*1.0, self.arena_size*1.0, world.dim_p)
        #         landmark.state.p_vel = np.zeros(world.dim_p)
        #
        #     landmark_dist = np.array([[np.linalg.norm(ag.state.p_pos - a.state.p_pos) for a in world.landmarks] for ag in world.landmarks])
        #     if np.all(np.min(landmark_dist,1) < 2.0) and np.all(np.max(landmark_dist,1) < 3.0):
        #         landmark_reach = False
        #         # print('111')
        #     else:
        #         landmark_reach = True
        #         # print('222')



        world.max_steps_episode = 50
        world.steps = 0
        self.unconnections = 0
        self.coverage_rate_sum =0
        world.dists = []

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def matrixPow(self, Matrix, n):
        if (type(Matrix) == list):
            Matrix = np.array(Matrix)
        if (n == 1):
            return Matrix
        else:
            return np.matmul(Matrix, self.matrixPow(Matrix, n - 1))


    def reward(self, agent, world):

        if agent.iden == 0:  # compute this only once when called with the first agent
            # each column represents distance of all agents from the respective landmark

            self.adj_matrix = np.array([[1 if np.linalg.norm(ag.state.p_pos - a.state.p_pos) < self.connection_distres else 0 for a in world.agents] for ag in world.agents])
            unit_matrix = np.identity(self.num_agents)
            D_matrix = (self.adj_matrix-unit_matrix).sum(1)
            las_matrix = D_matrix*unit_matrix - (self.adj_matrix-unit_matrix)
            eigenvalue, featurevector = np.linalg.eig(las_matrix)
            lambda_2 = np.sort(eigenvalue)[1] #

            if lambda_2 <= 0:
                connection_reward = -10
                self.unconnections += 1
            elif lambda_2 < 0.2:
                connection_reward = -1
            else:
                connection_reward = 0
            # print(agent.state.p_pos)
            world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
                                    for a in world.agents])
            # print(world.dists)
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = np.min(world.dists, 0) #
            coverage_sum = np.sum(self.min_dists <= self.dist_threshold)
            self.coverage_rate = coverage_sum/self.landmarks
            coverage_reward = 0
            self.coverage_rate_sum += self.coverage_rate

            # coverage_reward = coverage_sum/self.landmarks
            if world.steps == world.max_steps_episode:
                coverage_reward = (self.coverage_rate -1)*50 - world.steps*0.05

            if np.all(self.min_dists < world.dist_thres):
                coverage_reward = -world.steps*0.05

            joint_reward = -np.mean(np.clip(self.min_dists - self.dist_threshold, 0, 15))

            # print('111')
            self.rewards = np.full(self.num_agents,  (1-self.coverage_rate)*joint_reward*10 + 1*connection_reward)
            # self.rewards = np.full(self.num_agents,  (1-self.coverage_rate)*1*-1 + 1*connection_reward)

            world.min_dists = self.min_dists #

        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # print('111')
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        our_group = []
        for other in world.agents:
            if other is agent: continue
            our_group.append(other.state.p_pos - agent.state.p_pos)
            our_group.append(other.state.p_vel - agent.state.p_vel)
        default_obs = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + our_group + entity_pos)
        return default_obs

    def done(self, agent, world):
        condition1 = world.steps >= world.max_steps_episode
        # self.is_success = (self.unconnections < 1)
        self.is_success = np.all(self.min_dists < world.dist_thres) and (self.unconnections < 1)

        # self.is_success = np.all(self.min_dists < world.dist_thres)
        if world.landmarks[0].movable:
            return condition1
        else:
            return condition1 or self.is_success

    def info(self, agent, world):
        info = {'is_success': self.is_success, 'world_steps': world.steps,
                'reward': self.rewards.mean(), 'dists': self.min_dists.mean(),'unconnections': (self.unconnections > 0),'coverage_rate':self.coverage_rate,'coverage_rate_sum':self.coverage_rate_sum}
        return info
