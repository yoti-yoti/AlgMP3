import numpy as np
import os
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
from twoD.visualizer import Visualizer
from twoD.analysis_visualizer import plot_graph

# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}


def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    # execute plan
    for eps in [1, 10, 20]:
        plan = planner.plan(eps)
        print(f"eps: {eps}, path len: {planner.path_len}")
        DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.expanded_nodes, show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01, eta=1)

    # execute plan
    plan = planner.plan()
    print(f"path cost: {planner.compute_cost(plan)}")
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.2, k=None, step_size=None)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(
        bb=bb,
        start=MAP_DETAILS["start"],
        goal=MAP_DETAILS["goal"],
        ext_mode="E2",
        goal_prob=0.5,
        max_step_size=0.1,
    )
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    ## E2
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01, eta=0.1)
    # execute plan
    start = datetime.now()
    plan = planner.plan()
    total_time = datetime.now() - start
    print(f"E2 path cost: {planner.compute_cost(plan):.4f}, time: {total_time}")
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    ## E1
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.01, eta=0.1)
    # execute plan
    start = datetime.now()
    plan = planner.plan()
    total_time = datetime.now() - start
    print(f"E1 path cost: {planner.compute_cost(plan):.4f}, time: {total_time}")
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_loop():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    ## E2
    time_data = {
        0.05 : [],
        0.2  : []
    }
    cost_data = {
        0.05 : [],
        0.2  : []
    }
    for goal_prob in [0.05, 0.2]:
        for i in range(10):
            planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=goal_prob, eta=0.5)
            # execute plan
            start = datetime.now()
            plan = planner.plan()
            total_time = datetime.now() - start
            cost = planner.compute_cost(plan) 
            time_data[goal_prob].append(total_time.total_seconds())
            cost_data[goal_prob].append(cost)
            # Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    print(f"avg_t_5 = {sum(time_data[0.05])/10:.4f}")
    print(f"avg_t_20 = {sum(time_data[0.2])/10:.4f}")
    print(f"avg_c_5 = {sum(cost_data[0.05])/10:.4f}")
    print(f"avg_c_20 = {sum(cost_data[0.2])/10:.4f}")
    print(f"std_t_5 = {np.std(time_data[0.05]):.4f}")
    print(f"std_t_20 = {np.std(time_data[0.2]):.4f}")
    print(f"std_c_5 = {np.std(cost_data[0.05]):.4f}")
    print(f"std_c_20 = {np.std(cost_data[0.2]):.4f}")

    plot_graph(time_data, cost_data, 0.05, 0.2)


def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))

        visualizer.show_path(path)


if __name__ == "__main__":
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    # run_3d()
    # run_2d_rrt_loop()