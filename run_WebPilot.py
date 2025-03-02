
import json
import os
import re
import subprocess
import time
import logging
import traceback
import sys
import pandas as pd
import datetime

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# add webpilot_utils to the path
sys.path.append("./webpilot_utils/.")

from env_init import cookies_init, setting_url
# setting_url.setting_url("webarena") # uncomment this line if you want to set the url

from WebPilot.model import MAX_TERMINAL_NODES, MAX_ITERATION_TIMES

# set window size
WIDTH = 1600
HEIGHT = 1200

from env_init import gen_task  
gen_task.generate_task_config_files()

from agent.utils import Trajectory
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from evaluation_harness.evaluators import evaluator_router

from WebPilot.MCTS.WTS_cls import WTS_cls

from webpilot_utils.remove_generated_files import remove_generated_files_id

if not os.path.exists("./webpilot_utils/node_pics"):
    os.makedirs("./webpilot_utils/node_pics")

def setup_logger(task_id):
    if not os.path.exists(f"./task_info/{task_id}"):
        os.makedirs(f"./task_info/{task_id}")

    log_file_single_cycle = f"./task_info/{task_id}/{task_id}.log"
    open(log_file_single_cycle, "w").close()
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_single_cycle, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return

def single_task(TASK_ID, MAX_TERMINAL_NODES, MAX_ITERATION_TIMES):
    single_start_time = time.time()
    setup_logger(TASK_ID)

    # 1. Init the environment.
    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=100,
        observation_type="accessibility_tree", # html or image
        current_viewport_only=True,
        viewport_size={"width": WIDTH, "height": HEIGHT},
    )

    # 2. choose file and reset
    file_ID = TASK_ID
    config_file = f"webarena-main/config_files/{file_ID}.json"
    config_info = json.load(open(config_file, "r"))
    config_info_to_log = json.dumps(config_info, indent=4)
    logging.info(f"config_info: {config_info_to_log}")

    intent = config_info["intent"]
    trajectory: Trajectory = []


    # 3.1 init
    WTS = WTS_cls(intent, config_file, file_ID, env, MAX_TERMINAL_NODES, MAX_ITERATION_TIMES)

    # 3.2 execution (agent working)
    WTS.run()

    # 3.3 output the result
    [final_answer, trajectory, env] = WTS.final_output()
    trajectory.append(create_stop_action(repr(final_answer)))

    # 4. evaluation
    evaluator = evaluator_router(config_file)
    score = evaluator(
        trajectory=trajectory,
        config_file=config_file,
        page=env.page,
        client=env.get_page_client(env.page),
    )
    logging.info("===================================== Exploration finished =====================================")
    logging.info(f"score is: {score}")

    single_end_time = time.time()
    time_usage = round(single_end_time - single_start_time, 2)

    # 5. assert the result and close the env
    if score > 0:
        print("Success, the score is: ", score)
        logging.info(f"Success, the score is: {score}\n\n\n")
    else:
        print("Failed, please check the reason")
        logging.info("Failed, please check the reason\n\n\n")
    env.close()

    remove_generated_files_id(TASK_ID)

    # clean webpilot_utils/node_pics
    subprocess.run(["bash", "-c", "rm -rf ./webpilot_utils/node_pics/*"])

    return None

def run_list(list, max_terminal_nodes=1, max_iteration_times=15):
    for task_id in list:
        single_task(task_id, max_terminal_nodes, max_iteration_times)

def main():

    # get the input arguments of a list
    task_list = sys.argv[1:]

    if task_list:
        task_list = [int(i) for i in task_list]
        run_list(task_list, MAX_TERMINAL_NODES, MAX_ITERATION_TIMES)

if __name__ == "__main__":
    main()
