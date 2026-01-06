import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []
        self.path_len = 0

    def plan(self, eps = 1):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = eps
        plan = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        return self. epsilon * self.bb.compute_distance(state, self.goal)

    def expand_nodes(self, parent_map, curr):
        self.expanded_nodes = []
        self.expanded_nodes.append(curr)
        while curr in parent_map:
            curr = parent_map[curr]
            self.expanded_nodes.append(np.array(curr))
        self.expanded_nodes.reverse()
        return self.expanded_nodes

    def a_star(self, start_loc, goal_loc):
        start_node = tuple(start_loc)
        goal_node = tuple(goal_loc)
        openlist = []
        heapq.heappush(openlist, (0.0, 0.0, start_node))
        closedlist = set()
        gmap = {start_node: 0.0}
        parent_map = {}
        while openlist:
            f, g, q = heapq.heappop(openlist)
            qx, qy = q
            if q in closedlist:
                continue
            if q == goal_node:
                self.path_len = g
                return self.expand_nodes(parent_map, goal_node)
            closedlist.add(q)
            for dx, dy in self.directions:
                successor = (qx + dx, qy + dy)
                if self.bb.config_validity_checker(successor) and self.bb.edge_validity_checker(q, successor):
                    if successor in closedlist:
                        continue
                    g = gmap[q] + self.bb.compute_distance(q, successor)
                    if successor not in gmap or g < gmap[successor]:
                        parent_map[successor] = q
                        gmap[successor] = g
                        f = g + self.compute_heuristic(successor)
                        heapq.heappush(openlist, (f, g, successor))
    
