import time
import logging
import math
import pickle
import os
import shutil
import datetime
import copy

from WebPilot.model import (
    MCTS_VARIANT_FLAG,
    MAX_RECTIFY_TIMES,
)

from WebPilot.env_utils import *
from WebPilot.agent_func.all_agent_funcs import *

from WebPilot.mcts_vis import (
    MCTS_vis,
)

from .Node_cls import Node_cls

# ==================== MCTS_cls ====================


class MCTS_cls:
    # framwork
    def __init__(self, WebTask, SubTask, env, obs,
                 sample_num: int = 3, max_depth: int = 10,
                 last_terminal_node: Node_cls = None,
                 aggr_reward: str = "max", aggr_child: str = "max",
                 w_exp: float = 1) -> None:
        self.WebTask = WebTask
        self.SubTask = SubTask
        self.env = env
        self.actree = obs["text"]
        self.screenshot = obs["image"]
        self.sample_num = sample_num
        self.max_depth = max_depth
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child
        self.w_exp = w_exp
        self.rollout_times = 0
        self.terminal_nodes = []
        self.env_terminated = False
        self.stop_flag = False

        # initialize the root node
        self.root = Node_cls(0, self.actree, env.page.url, last_terminal_node, {
                             "action_type": "noop", "action_str": "None", "element_info": ""}, self.max_depth)
        self.root.name = f"root_{SubTask.idx+1}"
        self.root.save_img(self.screenshot)
        self.root.update_rapid_access()
        if last_terminal_node is not None:
            self.root.observation_description = last_terminal_node.observation_description
            self.root.node_reflection_for_child = ""

        # visualization initialization
        vis_filename = str(WebTask.task_id) + "_" + str(SubTask.idx+1)
        self.Graph = MCTS_vis(self.root, WebTask.content, SubTask.content,
                              SubTask.expectation, filename=vis_filename)
        self.Graph.view()

        # record the initial info of the WebTask for env_rollout_reset
        self.WebTask.initial_url = env.page.url

        # record the general_execution_reflections, currently for single MCTS
        self.execution_reflections = []

    def _env_step(self, src_action: dict):
        """
        control the env forward, used in expansion, simulation
        """
        action_to_execute = gen_executable_action(src_action)
        obs, _, _, _, info = self.env.step(action_to_execute)

        # will only be used in final output of MCTS_cls, only store the final step's info
        self.info_for_trajectory = {"observation": obs, "info": info, }
        
        self.actree = trimm_actree(obs["text"])
        self.screenshot = obs["image"]

        return obs

    def _env_step_with_verifier(self, src_action):
        """
        control the env forward, with an interact_verifier, to ensure the action is executable
        """
        former_actree = self.actree
        former_url = self.env.page.url
        former_gen_action_prompt = src_action["gen_action_prompt"]
        execution_succeed = False
        failed_execution_reflection = ""
        max_verifier_times = 3
        verifier_times = 0
        while not execution_succeed and verifier_times < max_verifier_times:
            self._env_step(src_action)
            new_actree = self.actree
            new_url = self.env.page.url
            execution_succeed, execution_reflection = ActionVerifier.interact_verifier(
                src_action, former_actree, former_url, new_actree, new_url,)

            if execution_succeed and failed_execution_reflection.strip() == "":
                return src_action
            elif not execution_succeed:
                failed_execution_reflection += execution_reflection
                failed_action = src_action
            elif execution_succeed and failed_execution_reflection.strip() != "":
                return src_action

            # the src_action is wrong, re-generate the action
            print_n_log("===== Current process: re_gen_action in _env_step_with_verifier =====")
            verifier_times += 1
            src_action = re_gen_action(
                former_gen_action_prompt, failed_execution_reflection, failed_action=src_action, actree=new_actree)
        # if the verifier_times is larger than the max_verifier_times
        if verifier_times >= max_verifier_times:
            print_n_log("===== Current process: re_gen_action reaches the max_verifier_times -> verifier failed =====")
        return src_action

    def _env_step_transfer_id(self, src_action: dict, log_real_action=False):
        """
        works for _env_back_by_rest(), for given new actree, find the corresponding element_id according to the element_nbh
        only transfer the action_id, but not changed the initial action
        log_real_action: only works for env_goto_node, to log the real action before and after transfer, for comparison and debugging
        """
        if log_real_action:
            before_str = f"""Action: {src_action["action_type"]} | {src_action["action_str"]} | {src_action["element_info"]}\n"""
        action_to_execute = action_id_transfer(src_action, self.actree)

        # log the action before and after transfer
        if log_real_action:
            after_str = f"""Action: {action_to_execute["action_type"]} | {action_to_execute["action_str"]} | {action_to_execute["element_info"]}\n"""
            log_str = f"===== action_id_transfer: ===== \n before transfer: {before_str}\n after transfer: {after_str} \n===== Check if the element_info are aligned. ====="
            logging.info(log_str)

            try:
                log_str += f"""===== nbh before transfer: {src_action["element_nbh"]} =====\n"""
                log_str += f"""===== nbh after transfer: {action_to_execute["element_nbh"]} =====\n"""
            except:
                pass
            # record the after_str and the actree where the action is executed in a single file for debugging
            # create a specific log file with special name according to the time
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            log_file_name = f"./action_transfer_logs/a_t_log_{self.WebTask.task_id}_{current_time}.log"
            # create if not exists
            if not os.path.exists("./action_transfer_logs"):
                os.makedirs("./action_transfer_logs")
            with open(log_file_name, "a") as f:
                f.write(log_str)
                f.write(f"===== The actree where the action is executed: =====\n{self.actree}\n\n\n")
            logging.info(f"===== The action transfer log is saved in: {log_file_name} =====")

        src_action = action_to_execute
        action_to_execute = gen_executable_action(src_action)
        obs, _, _, _, info = self.env.step(action_to_execute)
        self.actree = trimm_actree(obs["text"])
        self.screenshot = obs["image"]

        # store the final step's info
        self.info_for_trajectory = {"observation": obs, "info": info, }

        return info
    
    def _env_back_by_reset(self, node: Node_cls):
        """
        given a node, set the env back to its parent node
        2 steps: 1. reset the env to initial state(root of the current subtask); 2. execute the path to the parent node
        """
        self._env_rollout_reset()

        # get path to parent node
        path = []
        while not "root" in node.name:
            path.insert(0, node)
            node = node.parent
        # execute the path
        for node in path:
            self._env_step_transfer_id(node.src_action)

        return None

    def _env_goto_node(self, node: Node_cls):
        """
        here we want to got the node rapidly, works for env_back,
        also in selection part: we could just select node without env_step, but use this function to go to the node rapidly. However, this node must be already visited.
        Work flow: 1. get the last reachable url; 2. get the path; 3. execute the path
        The last reachable url and the path are stored in the node. A dict: rapid_access = {"last_url":str, "path":list[Node_cls]}
        """

        # record the former actree and url
        former_actree = node.actree
        former_url = node.url

        last_url = node.rapid_access["last_url"]
        path = copy.deepcopy(node.rapid_access["path"])

        # execute the path
        print_n_log(f"===== Current process: env_goto_node, going to the node: {node.name[:6]} =====")
        print_n_log(f"===== The last reachable url is: {last_url} =====")

        # add a intermediate url if the last_url is same as the env's current url, to erase the saved local storage
        if last_url == self.env.page.url:
            print_n_log("===== The last reachable url is the same as the current url, go to an empty page first. =====")
            current_file_dir = os.path.dirname(__file__)
            empty_page_url = os.path.join(current_file_dir, "empty_page.html")
            empty_page_url = "file://" + empty_page_url
            goto_empty_page_action = {"action_type": "goto", "action_str": f"{empty_page_url}"}
            self._env_step_transfer_id(goto_empty_page_action)

        goto_url_action = {"action_type": "goto", "action_str": f"{last_url}"}
        step_success_info = []
        info = self._env_step_transfer_id(goto_url_action)
        step_success_info.append(info)
        for action in path:
            info = self._env_step_transfer_id(action, log_real_action=False)
            step_success_info.append(info)

        print_n_log(f"===== env_goto_node, going to the node: {node.name[:6]} finished. The num of executed actions is 1+{len(path)} =====")
        logging.info(f"===== After env_goto_node, the current url is: {self.env.page.url} =====")
        node.actree = self.actree

        return None

    def _env_rollout_reset(self):
        """
        this function will reset the env every rollout, so the env starts from the root again.
        just use env_goto_node to go to the root node
        """
        print_n_log("========== Resetting the env for the rollout ......")
        self._env_goto_node(self.root)
        self.env_terminated = False
        print_n_log(f"========== Resetting finished ==========")

        return None

    def _rollout(self, root: Node_cls):
        # single rollout from the current node, normally the root node
        node = self._selection(root)

        # here add a if judgement to stop searching if the root node is terminal
        if "root" in node.name and node.terminal:
            self.stop_flag = True
            return None

        node = self._expansion(node)
        node = self._simulation(node)
        self._backpropagation(node)
        if not node.terminal: # only reset the env if the node is not terminal
            self._env_rollout_reset()

        # dump the root node every time for debugging & have a look on the reflections
        info_pkl = [self.root, self.WebTask, self.SubTask]
        pkl_file = f"root_{self.WebTask.task_id}.pkl"
        pickle.dump(info_pkl, open(pkl_file, "wb"))
        record_info(str(self.WebTask.task_id), str(
            self.SubTask.idx+1), picklefile_update=True)

    def _selection(self, node: Node_cls) -> Node_cls:
        """
        select the node to be expanded(if visited) or simulated(unvisited)
        Starting from root node every time.
        """
        print_n_log("===== Current process: selection =====")
        # set an empty executed_actions for each selection
        self.SubTask.executed_actions = []
        while True: 
            self.Graph.node_path(node)
            if not node.children:
                if not "root" in node.name:
                    self._env_goto_node(node)
                return node
            if all(child.terminal for child in node.children):
                node.terminal = True
                self.Graph.node_terminal(node)
                if "root" in node.name:
                    logging.info("===== All the children of the root node are terminal, stop the search in advance. =====")
                    node.terminal = True
                    return node
                node = node.parent
                continue
            uct_children = [
                child for child in node.children if not child.terminal]
            node = self._uct_select(uct_children)

            if node.src_action == "EMPTY ACTION":
                self._env_goto_node(node.parent)
                # generate a new src_action with the reflection generated by the sibling node
                parent_node_sim_reflection = node.parent.sim_reflection \
                    if hasattr(node, "parent") and hasattr(node.parent, "sim_reflection") else ""
                print_n_log("===== Current process: gen_action_with_reflection in selection =====")
                logging.info(f"===== Current Node name: {node.name[:4]} =====")
                logging.info(f"===== Current Node url: {node.url} =====")
                parent_not_stop_reason = node.parent.is_subtask_stopped_reason if hasattr(
                    node.parent, "is_subtask_stopped_reason") else ""
                parent_action = node.parent.src_action if hasattr(node.parent, "src_action") else ""
                parent_node_reflection_for_child = node.parent.node_reflection_for_child if hasattr(
                    node.parent, "node_reflection_for_child") else ""
                node.src_action = gen_action_with_reflection(actree=self.actree,
                                                             WebTask=self.WebTask,
                                                             SubTask=self.SubTask,
                                                             sibling_actions=node.get_sibling_actions(),
                                                             sibling_actions_n_reflections=node.get_sibling_actions_n_reflections(),
                                                             parent_node_sim_reflection=parent_node_sim_reflection,
                                                             parent_node_reflection_for_child=parent_node_reflection_for_child,
                                                             parent_not_stop_reason=parent_not_stop_reason,
                                                             parent_action=parent_action,
                                                             )
                # here check the output, if the delete_node signal received, then delete the node, where the generated action is still the same as its siblings
                if isinstance(node.src_action, str) and "DELETE_NODE" in node.src_action:
                    self.Graph.node_delete(node)
                    delete_node_name = node.name
                    node = node.parent
                    for child in node.children:
                        if child.name == delete_node_name:
                            node.children.remove(child)
                            logging.info(f"===== Node {delete_node_name} is deleted. Return to the parent node for selection. =====")
                            break
                    continue

                print_n_log("===== Current process: selection, a new node generated (with visited sibling), step into it with verifier =====")
                node.src_action = self._env_step_with_verifier(node.src_action)
                self.SubTask.executed_actions.append(node.src_action)
                node.url = self.env.page.url
                node.actree = self.actree
                node.save_img(self.screenshot)
                node.update_rapid_access()
                return node
            else:
                # currently not a new node,
                if not "root" in node.name:
                    self.SubTask.executed_actions.append(node.src_action)
                # focus on the child, and then step the env to the corresponding node(state), update the info of the node
                print_n_log(f"===== Current process: selection, found a visited node {node.name[:6]}, continue selection without directly env_step to save time until the leaf node =====")

    def _expansion(self, node: Node_cls) -> Node_cls:
        """
        expand the node, generate the children of the node
        """

        print_n_log("===== Current process: expansion =====")

        if node.depth >= self.max_depth:  # too deep
            node.terminal = True
            return node
        if node.visit_counts > 0: # >0 means the node visited before.
            node._gen_children_empty(self.WebTask, self.SubTask, self.sample_num)

            logging.info(f"===== Current Node name: {node.name[:4]} =====")
            logging.info(f"===== Current Node url: {node.url} =====")
            print_n_log(
                "===== Current process: gen_action in expansion without siblings' reflection (brand new node) =====")
            logging.info(
                f"===== The new node(to be expanded)'s name: {node.children[0].name[:4]} =====")
            logging.info(
                f"===== Its url is currently not available, since env is not stepped yet. =====")

            # judge whether the previous sim_node is scored with 10, if so, then the new node should be the same
            if hasattr(node, "sim_node_to_real_flag") and node.sim_node_to_real_flag:
                # choose the first child, since all unvisited
                node = node.children[0]
                node.src_action = node.parent.sim_node_to_real_action
                print_n_log(
                    "===== Current process: expansion: a brand new node(first child) generated, inherit the sim_action since it's scored with >9 =====")
                logging.info(f"The actree before inheriting the sim_action is : \n{self.actree}")
                logging.info("Check the action transferred for the inherited action")
                self._env_step_transfer_id(node.src_action, log_real_action=False)
            else:
                # get sim_reflection, but root node doesn't have one
                parent_node_sim_reflection = node.sim_reflection if hasattr(
                    node, "sim_reflection") else ""
                parent_node_stop_reason = node.is_subtask_stopped_reason if hasattr(
                    node, "is_subtask_stopped_reason") else ""
                parent_node_reflection_for_child = node.node_reflection_for_child if hasattr(
                    node, "node_reflection_for_child") else ""
                parent_action = node.src_action if hasattr(
                    node, "src_action") else ""
                src_action = gen_next_action(actree=node.actree,
                                             WebTask=self.WebTask,
                                             SubTask=self.SubTask,
                                             parent_node_sim_reflection=parent_node_sim_reflection, # the reflection of the current node, generated in the simulation part
                                             parent_node_reflection_for_child=parent_node_reflection_for_child,
                                             parent_not_stop_reason=parent_node_stop_reason,
                                             parent_action=parent_action,
                                             )
                node = node.children[0]
                node.src_action = src_action

                print_n_log(
                    "===== Current process: expansion: a brand new node(first child) generated, step into it with verifier =====")
                node.src_action = self._env_step_with_verifier(node.src_action)

            self.SubTask.executed_actions.append(node.src_action)  # added for further simulation
            node.actree = self.actree
            node.save_img(self.screenshot)
            node.url = self.env.page.url
            node.update_rapid_access()

        if not "root" in node.name:
            self.Graph.add_node(node)
        if not node.terminal:
            # set this node as the current focus node
            self.Graph.node_focus(node)

        return node  # return the node to be simulated

    def _simulation(self, node: Node_cls) -> Node_cls:
        """
        simulate only one step to align with method for WebArena
        """
        print_n_log("===== Current process: simulation =====")
        logging.info(f"===== Current Node name: {node.name[:4]} =====")
        logging.info(f"===== Current Node url: {node.url} =====")

        node.visit_counts += 1

        if "root" in node.name:  # root node doesn't need to be simulated
            return node

        # 1. if not terminal, the node to simulate must be visited the first time, here we ask the agent to describe the observation
        print_n_log(
            "===== Current process: gen_observation_description before next_step_simulation =====")
        url_changed = True if not is_same_base_url(node.url, node.parent.url) else False
        node.obs_des_dict = gen_observation_description(node.actree, node.parent.actree, self.WebTask, self.SubTask, node.src_action, url_changed)
        node.observation_description = node.obs_des_dict["description"]
        node.src_action["action_effect"] = node.obs_des_dict["changes"]
        node.src_action["intent_fulfillment"] = node.obs_des_dict["action_intent_fulfillment"]

        # 1.1 after observation, use the node_evalution to get a score for itself, the score will be used together with score from simulation to update the node's reward
        print_n_log(
            "===== Current process: node_evaluation before simulation part =====")
        logging.info(f"===== Current Node name: {node.name[:4]} =====")
        logging.info(f"===== Current Node url: {node.url} =====")
        node.score_reason_self, score_list = node_evaluation(
            self.actree, self.WebTask, self.SubTask, self.SubTask.executed_actions, node.obs_des_dict)
        # the first element of the score_list is the reward
        node.reward_self = score_list[0]
        # the rest elements are the separate rewards
        node.reward_separate = score_list[1:]

        # 2. after observation, generate a node_reflection, which is used for sibling node generation; This reflection is generated without evaluation
        logging.info(f"===== Current Node name: {node.name[:4]} =====")
        logging.info(f"===== Current Node url: {node.url} =====")
        obs_des_n_changes_n_fulfillment = f"**Description**: {node.obs_des_dict['description']}\n" \
            + "**Changes**: {node.obs_des_dict['changes']}\n" \
            + "**Action Intent Fulfillment**: {node.obs_des_dict['action_intent_fulfillment']}\n" \
            
        node.node_reflection_for_child, node.node_reflection_for_sib = node_reflection(actree=node.actree,
                                                                                       SubTask=self.SubTask,
                                                                                       WebTask=self.WebTask,
                                                                                       src_action=node.src_action,
                                                                                       obs_des_dict=node.obs_des_dict,)
        node.node_reflection = f"[Reflection for child]: {node.node_reflection_for_child}\n[Reflection for sim]: {node.node_reflection_for_sib}\n"

        # 3.1 simulation action generation 
        print_n_log(
            "===== Current process: SIMULATION!: gen_action_with_stop_asking in simulation part =====")
        logging.info(f"===== Current Node name: {node.name[:4]} =====")
        logging.info(f"===== Current Node url: {node.url} =====")

        is_subtask_stopped_reason, sim_action_dict, intermediate_info = \
            gen_action_with_stop_asking(actree=node.actree, WebTask=self.WebTask, SubTask=self.SubTask, node=node,)
        # could also be the reasoning_process from subtask_stop_verifier
        node.is_subtask_stopped_reason = is_subtask_stopped_reason

        # 3.2 action_type == "NONE" means the agent decide to stop
        if sim_action_dict["action_type"] == "NONE":  # means the agent decide to stop
            print(" STOP! ")
            if hasattr(self.SubTask, "need_answer") and self.SubTask.need_answer == True and \
                    intermediate_info.strip() != "":
                node.answer = intermediate_info
            node.terminal = True
            # no simulation, so directly assign the node.score_reason_self to node.score_reason, and node.reward_self to node.reward
            node.score_reason = node.score_reason_self
            node.reward = node.reward_self
            # evaluation and reflection directly
            logging.info(f"===== Current Node name: {node.name[:4]} =====")
            logging.info(f"===== Current Node url: {node.url} =====")
            node.M = node.Q = node.reward
            node.terminal_reflection = node.node_reflection
            if node.reward >= 5:
                self.Graph.node_terminal(node, valid_terminal=True)
                # only the high reward node will be added to the terminal_nodes
                self.terminal_nodes.append(node)

            # if this node and its siblings are all terminal, then the parent node should be terminal; this situation is also verified in selection part
            if all([child.terminal for child in node.parent.children]):
                node.parent.terminal = True
            return node  # no need to forward or backward

        # 3.3 Truely Simulation: if not stop, just step forward into the simulated node and get the evaluation
        print_n_log(
            "===== Current process: simulation: step into the simulated state with verifier =====")
        sim_action_dict = self._env_step_with_verifier(sim_action_dict)
        # save the sim_node img
        node.save_sim_img(self.screenshot)

        # generate the observation_description for the simulated state
        print_n_log(
            "===== Current process: gen_observation_description in simulated node (env already stepped into the simulated node) =====")
        logging.info(
            f"===== Current Node name (parent of simulated node): {node.name[:4]} =====")
        logging.info(
            f"===== Current Node's simulated CHILD url(not node itself): {self.env.page.url} =====")
        url_changed = True if not is_same_base_url(node.url, node.parent.url) else False
        sim_obs_des_dict = gen_observation_description(actree=self.actree, former_actree=node.actree,
                                        WebTask=self.WebTask, SubTask=self.SubTask, src_action=sim_action_dict,
                                        url_changed=url_changed)
        sim_action_dict["action_effect"] = sim_obs_des_dict["changes"]

        print_n_log(
            "===== Current process: node_evaluation in simulation part =====")
        logging.info(
            f"===== Current Node name (parent of simulated node): {node.name[:4]} =====")
        logging.info(
            f"===== Current Node's simulated CHILD url(not node itself): {self.env.page.url} =====")
        # add the simulated action to the executed_actions
        executed_actions = self.SubTask.executed_actions + [sim_action_dict,]
        node.score_reason_sim, score_list = node_evaluation(
            self.actree, self.WebTask, self.SubTask, executed_actions, sim_obs_des_dict,)
        node.reward_sim = score_list[0]
        node.reward_sim_separate = score_list[1:]

        # node.score_reason_self, node.reward_self
        # update the node's reward
        node.score_reason = f"[self score reason]:\n{node.score_reason_self}\n[sim score reason]:\n{node.score_reason_sim}"
        # 0.75 and 0.25 are the weights for self and sim respectively, for now.
        node.reward = (node.reward_self*0.75 + node.reward_sim*0.25)

        # 3.4 Here if the reward is 10, then we set this sim_node as a real node, since the agent is so confident about this action
        if node.reward_sim > 9:
            # make this sim_node to a real one, in the simulation part after gen_children_empty
            print_n_log(
                "After Evaluation of the simulated node, the score is 10. Set the simulated node to real")
            node.sim_node_to_real_flag = True
            node.sim_node_to_real_action = sim_action_dict

        # visualization of the simulated node
        self.Graph.node_simulation(node, sim_action_dict)

        # generate a node_reflection w.r.t both node.observation_description & node.score_reason
        print_n_log(
            "===== Current process: node_reflection in simulation part =====")
        logging.info(
            f"===== Current Node name (parent of simulated node): {node.name[:4]} =====")
        logging.info(
            f"===== Current Node's simulated CHILD url(not node itself): {self.env.page.url} =====")
        node.sim_reflection = sim_reflection(actree=self.actree,
                                             SubTask=self.SubTask,
                                             WebTask=self.WebTask,
                                             src_action=sim_action_dict,
                                             sim_obs_des_dict=sim_obs_des_dict,
                                             score=node.reward_sim,
                                             score_reason=node.score_reason_sim,)

        # here just use the reward, since simulation only happens once
        node.M = node.Q = node.reward
        return node

    def _backpropagation(self, node):
        """
        set the aggr_reward according to the aggr_reward method.
        if aggr_reward == "mean", introduce the discount_factor
        """
        print_n_log("===== Current process: backpropagation =====")

        if node.terminal:
            # self reflection (only for terminal nodes)
            print_n_log(
                "===== Current process: terminal_reflection in back_propagation part (agent decided to stop) =====")
            logging.info(f"===== Current Node name: {node.name[:4]} =====")
            logging.info(f"===== Current Node url: {node.url} =====")
            # node.terminal_reflection = terminal_reflection(node.actree, self.WebTask, self.SubTask, node.get_former_node_reflections(), node.score_reason)
            node.terminal_reflection = ""

        while not "root" in node.name: 
            self.Graph.node_update(node)  # update the node' value
            # reset the node to normal style if not terminal
            if not node.terminal:
                self.Graph.node_reset(node)
            reward_child = node.M
            node = node.parent
            node.visit_counts += 1
            node.M = max(node.M, reward_child)
            child_discount_factor = 0.9
            node.Q = (node.Q * (node.visit_counts-1) + reward_child *
                      child_discount_factor) / node.visit_counts  # mean
        self.Graph.node_reset(self.root)

        return None

    def _uct_select(self, nodes: list) -> Node_cls:
        # select the node with the highest UCT,
        uct_dict = {}
        # max or mean decide which method to aggregate the reward
        if self.aggr_child == "max":
            for node in nodes:
                uct_dict[node] = node.M + uct_explore(node, self.w_exp)
        elif self.aggr_child == "mean":
            for node in nodes:
                uct_dict[node] = node.Q + uct_explore(node, self.w_exp)
        return max(uct_dict, key=uct_dict.get)

    def set_node_with_max_value_to_terminal(self):
        """
        if the maximum rollout times reached, set the leaf node with the highest value to terminal
        """
        logging.info("Set the leaf node with the highest value to terminal.")
        leaf_nodes = []

        def dfs(node):
            # if node doesn't have children, then it is a leaf node
            if not node.children and node.src_action != "EMPTY ACTION":
                leaf_nodes.append(node)
            # if node has children, but all children's src_action = "EMPTY ACTION", then it is a leaf node
            elif node.children != [] \
                    and all([child.src_action == "EMPTY ACTION" for child in node.children]):
                leaf_nodes.append(node)
            else:
                for child in node.children:
                    dfs(child)

        # construct the leaf_nodes list
        dfs(self.root)
        max_value_node = max(leaf_nodes, key=(lambda x: x.reward))
        target_node_name = max_value_node.name

        # compare the node name with the nodes in the root node, set the node with the same name to terminal
        def dfs_set_terminal(node):
            if node.name == target_node_name:
                logging.info(
                    f"Node with the highest value is set to terminal.\n Node name: {node.name[:4]} Node reward: {node.reward}")
                node.terminal = True
                # this node with the highest value will be set to valid_terminal
                self.Graph.node_terminal(node, valid_terminal=True)
                node.terminal_reflection = node.node_reflection
                self.terminal_nodes.append(node)
                return None
            for child in node.children:
                dfs_set_terminal(child)

        dfs_set_terminal(self.root)

    # return the best trajectory

    def _best_path(self) -> list:
        """
        select the best path based on some criteria 
        """
        paths = self._terminal_paths()

        best_criteria = "M"  # "Q" or "M" or "num_steps"
        if best_criteria == "Q":
            best_idx = max(range(len(paths)), key=(lambda i: paths[i][-1].Q))
        elif best_criteria == "M":
            best_idx = max(range(len(paths)), key=(lambda i: paths[i][-1].M))
        elif best_criteria == "num_steps":
            pass  
        best_path = paths[best_idx]

        return best_path

    def _terminal_paths(self) -> list:
        """
        get all paths from all terminal nodes. generated path start from root to the terminal leaf node
        paths is list of path; path is list of nodes
        """
        paths = []
        if self.terminal_nodes == []:
            self.terminal_nodes.append(self.root)

        for node in self.terminal_nodes:
            path = []
            while not "root" in node.name:  # root is not included
                path.insert(0, node)
                node = node.parent

            path.insert(0, self.root)
            paths.append(path)

        return paths

    def _get_terminal_reflections(self) -> str:
        """
        get all terminal reflections from all terminal nodes
        """
        reflections = []
        idx = 0
        for node in self.terminal_nodes:
            idx += 1
            # add a number before each reflection
            reflections.append(str(idx)+". "+node.terminal_reflection)
        reflections = "".join(reflections)
        return reflections

    # process to run the MCTS algorithm
    def run(self, max_terimal_nodes: int = 1, max_rollout_times: int = 20):
        """
        this function constructs the tree of the SubTask
        stop loop according to: 1. counts of terminal nodes; 2. counts of rollout times
        """
        while len(self.terminal_nodes) < max_terimal_nodes and self.rollout_times < max_rollout_times:
            print("====== singel rollout start ======")
            print(f"Terminal nodes: {len(self.terminal_nodes)}")
            print(f"Rollout times: {self.rollout_times}")
            self._rollout(self.root)
            if self.stop_flag:
                break
            print("====== single rollout finished ======")
            self.Graph.view()
            self.rollout_times += 1

        # if terminal_nodes reached the max_terminal_nodes number
        if len(self.terminal_nodes) >= max_terimal_nodes:
            logging.info("Max terminal nodes reached, stop the MCTS algorithm.")

        # if rollout_times reached the max_rollout_times, consider using the leaf node with the highest M value
        if self.rollout_times >= max_rollout_times:
            logging.info("Max rollout times reached, stop the MCTS algorithm.")
            # if no terminal nodes in the terminal_node list ,then set the node with the highest value to terminal
            if len(self.terminal_nodes) == 0:
                self.set_node_with_max_value_to_terminal()
                self._env_goto_node(self.terminal_nodes[0]) # go to the terminal node with the highest value

        # tree construction finished
        self.stop_flag = self.SubTask.stop_flag = True

        # export root for recording
        info_pkl = [self.root, self.WebTask, self.SubTask]
        pkl_file = f"root_{self.WebTask.task_id}.pkl"
        pickle.dump(info_pkl, open(pkl_file, "wb"))

        return None

    def final_output(self):
        """
        excute the optimal_trajectory and return the result env
        output updated WebTask and observation
        information to be updated: WebTask.finished_subtasks, WebTask.executed_actions, WebTask.step_sum, observation
        control the env to run according to the final path
        """

        # get the path and updat the info of the SubTask
        path = self._best_path()

        # set the final node to final_terminal in the MCTS Graph
        self.Graph.node_final_terminal(path[-1])

        # consturct the executed_actions according to the path, root node is excluded
        self.SubTask.executed_actions = [node.src_action for node in path[1:]]

        # get the reflections_for_sib of the path[1:]
        reflections_for_sib = [
            node.node_reflection_for_sib for node in path[1:]]
        reflections_for_sib_str = ""
        for i, reflection in enumerate(reflections_for_sib):
            reflections_for_sib_str += f"{i+1}. {reflection}\n"

        # if the final terminal node has children, then it is forced to be terminal
        forced_terminal_flag = True if hasattr(
            path[-1], "reward_sim") else False

        logging.info(
            f"===== Current process: subtask completeness estimator in final_output part of subtask {self.SubTask.idx+1} =====")
        logging.info(f"===== Current Node name: {path[-1].name[:4]} =====")
        logging.info(f"===== Current Node url: {path[-1].url} =====")

        self.SubTask.subtask_completeness, self.SubTask.complete_flag, self.SubTask.subtask_reflection = \
            subtask_completeness_estimator(WebTask=self.WebTask, SubTask=self.SubTask, root_actree=path[0].actree, actree=path[-1].actree,
                                           reflections_for_sib=reflections_for_sib_str,
                                           forced_terminal_flag=forced_terminal_flag,)
        
        # judge whether the rectify time exceeds the max_rectify_times
        if self.SubTask.rectify_times >= MAX_RECTIFY_TIMES:
            self.SubTask.complete_flag = True # force set the complete_flag to True
            print_n_log(f"Rectify times reached the limit ({MAX_RECTIFY_TIMES}), force set the complete_flag to True.")
            # the rectify_times will be updated outside the function in the WTS loop

        if self.SubTask.complete_flag == True:
            trajectory = []
            
            subtask_final_terminal_node = path[-1]
            # self._env_goto_node(subtask_final_terminal_node) # no need to goto the final node, since the env should be already in the final state
            action_to_execute = action_id_transfer(subtask_final_terminal_node.src_action,
                                                   subtask_final_terminal_node.parent.actree)
            trajectory.append(action_to_execute)
            state_info = self.info_for_trajectory
            trajectory.append(state_info)

            # record the final actree for next subtask
            current_obs = state_info["observation"]
            current_obs["text"] = trimm_actree(current_obs["text"])
            current_actree = current_obs["text"]

            # will be saved when intermediate_info if generated
            final_answer = path[-1].answer

            # update the SubTask
            self.SubTask.steps = len(path)  # exclude the root node
            self.SubTask.trajectory = trajectory
            self.SubTask.executed_actions = [
                node.src_action for node in path]  # exclude the root node
            self.SubTask.final_answer = final_answer

            # update the WebTask
            self.WebTask.finished_subtasks.append(self.SubTask)
            self.WebTask.step_sum += self.SubTask.steps
            self.WebTask.executed_actions.extend(self.SubTask.executed_actions[1:])
            self.WebTask.trajectory.extend(trajectory)

            logging_finishing_subtask_info(
                path[-1], self.SubTask.final_answer, self.env.page.url, current_actree, self.SubTask.executed_actions)
            detail_observation_description = self.SubTask.detail_observation_description
            node_reflection_for_child = path[-1].node_reflection_for_child
            output_node = path[-1]

        else:
            self._env_goto_node(self.root)
            state_info = self.info_for_trajectory  # saved in env_step_transfer_id
            current_obs = state_info["observation"]
            current_obs["text"] = trimm_actree(current_obs["text"])
            detail_observation_description = ""
            node_reflection_for_child = ""
            output_node = self.root

        # record_info again
        info_pkl = [self.root, self.WebTask, self.SubTask]
        pkl_file = f"root_{self.WebTask.task_id}.pkl"
        pickle.dump(info_pkl, open(pkl_file, "wb"))
        record_info(str(self.WebTask.task_id), str(
            self.SubTask.idx+1), picklefile_update=True)

        env = self.env
        return env, current_obs, self.WebTask, self.SubTask, output_node


# ==================== uct selection function ====================
def uct_explore(node: Node_cls, w_exp: float = 1) -> float:
    """
    UCB function for selection phase
    """

    if MCTS_VARIANT_FLAG:
        # node_probability = 1 # set to 1 for now, regarding to the prior probability of the node(src_action)
        node_probability = 5
        # use the variant of UCT
        value = w_exp * node_probability * \
            math.sqrt(node.parent.visit_counts) / \
            (node.visit_counts + 1)  # function from AlphaZero
    else:
        # use the original UCT
        value = w_exp * math.sqrt(math.log(node.parent.visit_counts) / node.visit_counts) \
            if node.visit_counts > 0 else float("inf")  # if unvisited, set the value to inf

    return value

# ==================== record node information ====================
"""
This function will record the nodes' information for a single task. Input will be only a root node and task name.
"""
def record_info(task_id, subtask_idx, picklefile_update=False):
    """
    For a single task, record the info w.r.t each node.
    A node's items: node_name, src_action, action_intent, obs_description, score, score_reason, reflection(possible)
    """

    # archiv the pickle file and load the root node
    # create the path if not exist
    if not os.path.exists(f"task_info/{task_id}/pkls"):
        os.makedirs(f"task_info/{task_id}/pkls")
    # only when pickle_file_update is True, update the pickle file and pdf, otherwise use the original one
    if picklefile_update:
        shutil.copy(f"root_{task_id}.pkl", f"task_info/{task_id}/pkls/{task_id}_{subtask_idx}.pkl")
        logging.info("The root node pickle file doesn't exist. Maybe the task is info_extractor task.")
    info_pkl = pickle.load(
        open(f"task_info/{task_id}/pkls/{task_id}_{subtask_idx}.pkl", "rb"))
    root = info_pkl[0]
    # WebTask = info_pkl[1]
    # SubTask = info_pkl[2]

    def info_single_node(node: Node_cls) -> dict:
        info_dict = {}
        # info_dict["task"] = task_id
        info_dict["node_name"] = node.name[:4]
        info_dict["parent"] = node.parent.name[:4] if node.parent else ""
        # info_dict["src_action"] = node.src_action
        info_dict["src_action"] = repr(node.src_action["action_type"])+repr(node.src_action["action_str"]) \
            if isinstance(node.src_action, dict) else ""
        info_dict["src_action"] += "\n" + repr(node.src_action["element_info"]) \
            if isinstance(node.src_action, dict) and "element_info" in node.src_action else ""
        info_dict["element_nbh"] = node.src_action["element_nbh"] \
            if isinstance(node.src_action, dict) and "element_nbh" in node.src_action else ""
        info_dict["action_intent"] = node.src_action["action_intent"] \
            if isinstance(node.src_action, dict) and "action_intent" in node.src_action else ""
        info_dict["observation_description"] = node.observation_description \
            if hasattr(node, "observation_description") else ""
        info_dict["M-score"] = node.M
        info_dict["Q-score"] = node.Q
        info_dict["score"] = node.reward if hasattr(node, "reward") else None
        info_dict["score_reason"] = node.score_reason if hasattr(
            node, "score_reason") else ""
        info_dict["is_subtask_stopped_reason"] = node.is_subtask_stopped_reason \
            if hasattr(node, "is_subtask_stopped_reason") else ""
        info_dict["node_reflection"] = node.node_reflection \
            if hasattr(node, "node_reflection") else ""
        info_dict["sim_reflection"] = node.sim_reflection \
            if hasattr(node, "sim_reflection") else ""
        info_dict["terminal_reflection"] = node.terminal_reflection \
            if hasattr(node, "terminal_reflection") else ""
        return info_dict

    def info_all_nodes(node: Node_cls) -> list:
        info_list = []
        single_node = info_single_node(node)
        if single_node:
            info_list.append(info_single_node(node))
        for child in node.children:
            info_list.extend(info_all_nodes(child))
        return info_list

    # collect all info
    info_list = info_all_nodes(root)

    # convert to dataframe
    import pandas as pd
    df = pd.DataFrame(info_list)

    # move the current pdf to task_info folder
    if picklefile_update:
        # create the path if not exist
        if not os.path.exists(f"task_info/{task_id}/gvs"):
            os.makedirs(f"task_info/{task_id}/gvs")

        shutil.copy(f"{task_id}_{subtask_idx}.gv",
                    f"task_info/{task_id}/gvs/{task_id}_{subtask_idx}.gv")
        shutil.copy(f"{task_id}_{subtask_idx}.gv.pdf",
                    f"task_info/{task_id}/{task_id}_{subtask_idx}.pdf")

    # export to html
    if not os.path.exists(f"task_info/{task_id}/htmls"):
        os.makedirs(f"task_info/{task_id}/htmls")
    df.to_html(f"task_info/{task_id}/htmls/{task_id}_{subtask_idx}.html")

    # # transfer the screenshots # 12. March. mcts_vis url is hard to modify
    # check if the folder exists, if not, create it
    if not os.path.exists(f"task_info/{task_id}/node_pics"):
        os.makedirs(f"task_info/{task_id}/node_pics")

    # tranfer pics function, using pdf
    def transfer_screenshots(node):
        # check if pdf exists
        if os.path.exists(f"webpilot_utils/node_pics/{node.name}.pdf"):
            if "root" in node.name:
                shutil.copy(
                    f"webpilot_utils/node_pics/{node.name}.pdf", f"task_info/{task_id}/node_pics/{node.name}.pdf")
            else:
                shutil.copy(
                    f"webpilot_utils/node_pics/{node.name}.pdf", f"task_info/{task_id}/node_pics/{node.name}.pdf")

        # check for simulation node
        if os.path.exists(f"webpilot_utils/node_pics/{node.name}_sim.pdf"):
            shutil.copy(f"webpilot_utils/node_pics/{node.name}_sim.pdf",
                        f"task_info/{task_id}/node_pics/{node.name}_sim.pdf")

        for child in node.children:
            transfer_screenshots(child)

    transfer_screenshots(root)


def rename_incomplete_subtasks_files(task_id, subtask_idx):

    base_folder = f"task_info/{task_id}/"
    
    # create path if not exist
    for folder in ["pkls", "htmls", "gvs", "node_pics"]:
        if not os.path.exists(base_folder + folder):
            os.makedirs(base_folder + folder)

    base_name = f"{task_id}_{subtask_idx}"
    file_list = [f"{base_name}.pdf", 
                 f"pkls/{base_name}.pkl",
                 f"htmls/{base_name}.html", 
                 f"gvs/{base_name}.gv",
                 f"node_pics/root_{subtask_idx}.pdf",
                 ]
    # add base_folder to the file_list
    file_list = [base_folder + file_name for file_name in file_list]

    # get a time suffix
    timestamp = time.strftime("%H%M")

    # rename
    for file_name in file_list[:-1]:
        # replace the base_name with the new_name
        new_name = file_name.replace(base_name, f"{base_name}_{timestamp}")

        try:
            os.rename(file_name, new_name)
            print_n_log(f"File {file_name} renamed to {new_name}")
        except:
            logging.info(f"possible info_extraction task")
    
    # rename the last file
    try:
        root_pdf_name = file_list[-1].replace(subtask_idx, f"{subtask_idx}_{timestamp}")
        os.rename(file_list[-1], root_pdf_name)
        print_n_log(f"File {file_list[-1]} renamed to {root_pdf_name}")
    except:
        print_n_log("Root file possible not found")

    return None
