import time
import logging
import copy

from WebPilot.model import (
    MAX_SUBTASK_NUM,
    MAX_RECTIFY_TIMES,
    W_EXP,
    BRANCH_NUM,
)

from WebPilot.model import ask_LLM

from WebPilot.env_utils import (
    action_id_transfer,
    logging_evaluation_info,
    print_n_log,
)

from .WebTask_cls import WebTask_cls
from .SubTask_cls import SubTask_cls
from .MCTS_cls import MCTS_cls, record_info
from .Info_extractor_cls import Info_extractor_cls


# ==================== WTS_cls ====================
# WTS_cls is the class for a single web task
class WTS_cls:
    def __init__(self, intent, config_file, task_id, env, max_terminal_nodes=3, max_rollout_times=20):
        """
        initialization of the WTS class. 1. reset the env; 2. get the task content; 3. instantiate the WebTask_cls;
        4. IMPORTANT: copy a new env for testing, without influence on the original one
        """
        # logging the start of a single task
        logging.info(
            "\n\n\n======================================== single task start ========================================")
        for _ in range(3):
            logging.info(
                "============================================================================================")
        logging.info("Current task is: " + intent)

        self.env = env
        reset_start_time = time.time()
        obs, info = self.env.reset(options={"config_file": config_file})
        reset_end_time = time.time()
        logging.info(
            f"Reset the env for the task, cost time: {reset_end_time - reset_start_time} s")
        state_info = {"observation": obs, "info": info}

        print(f"========== Starting url: {self.env.page.url} ==========")

        # instantiate the WebTask
        self.WebTask = WebTask_cls(intent, task_id, config_file, obs["text"])
        self.WebTask.trajectory.append(state_info)

        # global settings
        global MAX_TERMINAL_NODES
        global MAX_ROLLOUT_TIMES
        MAX_TERMINAL_NODES = max_terminal_nodes
        MAX_ROLLOUT_TIMES = max_rollout_times

    def run(self):
        """
        the main loop of Web-Tree-Search
        find the best path of the miniwob task
        """
        WebTask = self.WebTask
        current_obs = WebTask.trajectory[0]["observation"]
        last_terminal_node = None  # the first subtask doesn't have a last_terminal_node
        actual_subtask_times = 0
        rectify_times = 0

        # high level loop
        while WebTask.stop_flag == False:
            if len(WebTask.subtasks) >= MAX_SUBTASK_NUM:
                print("Subtask number reached the limit, stop the task.")
                raise ValueError("Subtask number reached the limit, stop the task.")

            # whether the task needs SubTask decomposition
            if WebTask.decompose_flag:
                SubTask = WebTask.gen_subtask_from_plan_first(current_obs["text"])
                SubTask.rectify_times = rectify_times
            else:
                SubTask = SubTask_cls(content=WebTask.content, idx=0)
                SubTask.need_answer = WebTask.need_answer
                SubTask.expectation = WebTask.expectation

            actual_subtask_times += 1

            while SubTask.stop_flag == False:
                print_n_log(
                    "======================================== SubTask start ========================================")
                # use deepcopy to avoid the change of the node's info
                second_last_terminal_node = copy.deepcopy(last_terminal_node)
                match SubTask.interaction_type:
                    case "info_extraction":
                        Info_extractor_SubTask = Info_extractor_cls(
                            WebTask, SubTask, self.env, current_obs, max_depth=5, last_terminal_node=last_terminal_node)
                        Info_extractor_SubTask.run()
                        self.env, current_obs, WebTask, SubTask, last_terminal_node = Info_extractor_SubTask.final_output()
                    case "web_interaction":
                        MCTS_SubTask = MCTS_cls(WebTask, SubTask, self.env, current_obs, w_exp=W_EXP,
                                                sample_num=BRANCH_NUM, max_depth=7, last_terminal_node=last_terminal_node)
                        # using Global settings from WTS_cls
                        MCTS_SubTask.run(max_terimal_nodes=MAX_TERMINAL_NODES, max_rollout_times=MAX_ROLLOUT_TIMES)
                        self.env, current_obs, WebTask, SubTask, last_terminal_node = MCTS_SubTask.final_output()

                if SubTask.complete_flag == False:
                    last_terminal_node = copy.deepcopy(second_last_terminal_node)
                    rectify_times += 1
                else:
                    rectify_times = 0 # reset the rectify times for the next subtask

                if SubTask.interaction_type == "web_interaction":
                    record_info(str(WebTask.task_id), str(SubTask.idx+1), picklefile_update=True)
                self.observation = current_obs
            print_n_log(
                f"======================================== SubTask end, index: {SubTask.idx+1} ========================================")
            print_n_log(
                f"SubTask {SubTask.idx+1} finished, index: {SubTask.idx+1}, content: {SubTask.content}")
            if SubTask.complete_flag == False:
                print_n_log(
                    f"SubTask {SubTask.idx+1} is tested not completed, back to the root node state.")

            if WebTask.decompose_flag:
                # whether the main task is finished (inside the while loop & outside the subtask loop)
                WebTask.whether_stop_update_plan(final_actree=current_obs["text"],
                                                 final_subtask=SubTask,)
                
            else:
                # the task doesn't need to be composed, the subtask is the main task
                WebTask.stop_flag = True if SubTask.complete_flag else False
                if WebTask.need_answer:
                    WebTask.generate_final_answer(current_obs["text"])

        # end While loop
        # WebTask.generate_final_answer(current_obs["text"])
        self.WebTask = WebTask
        self.final_path = self.WebTask.executed_actions

        # record the info for evaluation for our own inspection
        logging_evaluation_info(final_answer=self.WebTask.final_answer,
                                final_url=self.env.page.url,
                                final_actree=current_obs["text"],
                                all_executed_actions=self.final_path,)

        return None

    def final_output(self):
        """
        search process is finished, run the original env according to the final path which is generated above
        construct the final answer and the trajectory
        """
        logging.info("===== WTS Final output =====")

        # this direct_output_flag, will skip the re-run of the final path
        direct_output_flag = True
        if direct_output_flag:
            return [self.WebTask.final_answer, self.WebTask.trajectory, self.env]

        # reset the env and run the final path
        obs, info = self.env.reset(
            options={"config_file": self.WebTask.config_file})

        trajectory = [{"observation": obs, "info": info}]

        webtask_final_terminal_node = self.final_path[-1]
        # goto final terminal node
        action_to_execute = action_id_transfer(webtask_final_terminal_node.src_action,
                                               webtask_final_terminal_node.parent.actree)
        trajectory.append(action_to_execute)
        state_info = self.info_for_trajectory
        trajectory.append(state_info)

        self.WebTask.trajectory = trajectory

        # record the info for evaluation for our own inspection
        logging_evaluation_info(final_answer=self.WebTask.final_answer,
                                final_url=self.env.page.url,
                                final_actree=obs["text"],
                                all_executed_actions=self.final_path,)

        return [self.WebTask.final_answer, self.WebTask.trajectory, self.env]
