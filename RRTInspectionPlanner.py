import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage, eta=0.2):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        self.eta = eta
        self.goal = np.array([0.3, 0.15, 1.0, 1.1]) ## for special case

        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.add_vertex(self.start, np.array([]))
        while self.tree.max_coverage < self.coverage:
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            self.extend(self.tree.get_nearest_config(rand_config), rand_config)
        
        # find shortest path
        plan = [np.array(self.tree.vertices[self.tree.max_coverage_id].config)]
        curr = self.tree.max_coverage_id
        while curr != 0:
            curr = self.tree.edges[curr]
            plan.append(np.array(self.tree.vertices[curr].config))
        plan.reverse()
        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        return self.bb.compute_path_cost(plan)

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.ext_mode=="E1":
            self.extend_1(near_config, rand_config)
        if self.ext_mode=="E2":
            self.extend_2(near_config, rand_config)

    def extend_1(self, near_config, rand_config):
        if self.bb.config_validity_checker(np.array(rand_config)) and self.bb.edge_validity_checker(np.array(near_config[1]), np.array(rand_config)):
            vid = self.tree.add_vertex(rand_config, self.bb.compute_union_of_points(self.bb.get_inspected_points(rand_config), self.tree.vertices[near_config[0]].inspected_points))
            self.tree.add_edge(near_config[0], vid, edge_cost=self.bb.compute_distance(near_config[1], rand_config))

    def extend_2(self, near_config, rand_config):
        # length = np.sqrt(sum([(near_config[i] - rand_config[i])**2 for i in range(len(near_config))]))
        length = self.bb.compute_distance(near_config[1], rand_config)
        if length == 0:
            return
        extend_config = [(near_config[1][i]) + min(self.eta, length) * ((rand_config[i]-near_config[1][i])/length) for i in range(len(near_config[1]))]
        if self.bb.config_validity_checker(np.array(extend_config)) and self.bb.edge_validity_checker(np.array(near_config[1]), np.array(extend_config)):
            vid = self.tree.add_vertex(extend_config, self.bb.compute_union_of_points(self.bb.get_inspected_points(extend_config), self.tree.vertices[near_config[0]].inspected_points))
            self.tree.add_edge(near_config[0], vid, edge_cost=min(self.eta, length))

