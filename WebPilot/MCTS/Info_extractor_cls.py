import logging
import json
import shutil

from WebPilot.model import ask_LLM

# import other functions defined by ourselves
from WebPilot.prompter.extractor import gen_info_extraction_prompt

from WebPilot.env_utils import *
from WebPilot.mcts_vis import Info_extractor_vis
  
from .MCTS_cls import MCTS_cls
from .Node_cls import Node_cls     

# ==================== info_extractor_cls ====================
class Info_extractor_cls(MCTS_cls):
    """
    info_extractor_cls is in the same level of MCTS_cls
    If the generated subtask is decided to be an "info_extraction" task, then the info_extractor_cls will be used instead of MCTS_cls
    """
    def __init__(self, WebTask, SubTask, env, obs, max_depth:int=5,
                last_terminal_node:Node_cls=None,) -> None:
        self.WebTask = WebTask
        self.SubTask = SubTask
        self.env = env
        self.obs = obs
        self.actree = obs["text"]
        self.screenshot = obs["image"]
        self.max_depth = max_depth

        self.root = Node_cls(0, self.actree, env.page.url, last_terminal_node,
                            {"action_type":"noop", "action_str":"None", "element_info":""}, self.max_depth)
        self.root.name = f"root_{SubTask.idx+1}"
        self.root.update_rapid_access()
        if last_terminal_node is not None:
            self.root.observation_description = last_terminal_node.observation_description
        else:
            self.root.observation_description = ""
        self.root.node_reflection_for_child = ""

        self.root.save_img(self.screenshot)

        # visualization initialization
        vis_filename = str(WebTask.task_id) + "_" + str(SubTask.idx+1)
        self.Graph = Info_extractor_vis(self.root, WebTask.content, SubTask.content, SubTask.expectation, filename=vis_filename)
        self.Graph.view()

        transfer_screenshot(self.root, WebTask.task_id, SubTask.idx+1)

    def get_answer(self, node:Node_cls):
        """
        ask whether the answer can be extracted from the current state
        Output: answer, answer_found, reason_for_not_stop
        """

        observation_description = ""

        executed_scroll_actions_str = ""
        for action in self.executed_scroll_actions:
            executed_scroll_actions_str += f"scroll {action['action_str']} \n"

        reflection_str = ""
        if hasattr(node.parent, "scroll_reason"):
            reflection_str = node.parent.scroll_reason

        finished_subtasks_str = ""
        for idx, subtask in enumerate(self.WebTask.finished_subtasks):
            finished_subtasks_str += f"{idx+1}. {subtask.content}\n"

        prompt = gen_info_extraction_prompt(task_content=self.SubTask.content, domain=self.WebTask.domain,
                                                expectation=self.SubTask.expectation, actree=node.actree,
                                                observation_description=observation_description,
                                                finished_subtasks_str=finished_subtasks_str,
                                                executed_scroll_actions=executed_scroll_actions_str,
                                                reflection=reflection_str,
                                                )                                       
        reponse = ask_LLM(prompt)
        answer_dict = json.loads(reponse)
        stop_flag = str2bool(answer_dict.get("stop_flag", False))
        reasoning_of_answer = answer_dict.get("reasoning_of_answer", "")
        answer = answer_dict.get("answer", "")
        scroll_reason = answer_dict.get("scroll_reason", "")
        scroll_flag = str2bool(answer_dict.get("scroll_flag", False))
        scroll_direction = answer_dict.get("scroll_direction", "down")
        not_feasible_reason = answer_dict.get("reason_for_not_feasible", "")
        not_feasible = False # currently not used

        node.scroll_reason = scroll_reason

        return stop_flag, answer, scroll_flag, scroll_direction, not_feasible_reason, not_feasible

    def scroll(self, node:Node_cls, scroll_direction:str) -> Node_cls:
        logging.info(f"=========== Scrolling {scroll_direction} to get more information. ===========")
        scroll_action = {"action_type":"scroll", "action_str":scroll_direction, "element_info":""}
        self.executed_scroll_actions.append(scroll_action)
        self.obs = self._env_step(scroll_action)
        child = Node_cls(node.depth+1, self.actree, self.env.page.url, node, scroll_action, self.max_depth)
        child.save_img(self.screenshot)
        node.children.append(child)
        self.Graph.add_node(child)
        self.Graph.view()
        transfer_screenshot(self.root, self.WebTask.task_id, self.SubTask.idx+1)
        return child

    def run(self):
        logging.info("========== Info extraction task starts. ==========")
        stop_flag = False
        scroll_times = 0
        node = self.root
        self.executed_scroll_actions = []
        # when answer is found or the max_depth is reached, stop the loop
        while scroll_times <= self.max_depth:
            stop_flag, answer, scroll_flag, scroll_direction, not_feasible_reason, not_feasible = self.get_answer(node)
            if stop_flag:
                logging.info("========== Answer found. ==========")
                logging.info(f"The answer is: {answer}")
                self.final_terminal_node = node
                self.SubTask.complete_flag = True
                self.SubTask.final_answer = answer
                self.SubTask.subtask_reflection = self.SubTask.subtask_completeness = \
                        f"The answer is found. \nAnswer: {answer}"
                self.Graph.node_terminal(node, answer)
                break
            elif scroll_flag:
                if scroll_times == self.max_depth:
                        logging.info("========== Max depth reached, but answer not found. ==========")
                        # max_depth reached, but answer not found
                        self.SubTask.complete_flag = False
                        self.SubTask.final_answer = "N/A"
                        self.SubTask.subtask_reflection = self.SubTask.subtask_completeness = \
                            "The answer is not found after scrolling to the max depth. Maybe I am not in the right page. Consider refine the plan."
                        break
                node = self.scroll(node, scroll_direction)
                scroll_times += 1
            elif not_feasible:
                logging.info("========== Current page is not enough, need further actions ==========")
                self.SubTask.complete_flag = False
                self.SubTask.final_answer = ""
                self.SubTask.subtask_reflection = self.SubTask.subtask_completeness = \
                        f"{not_feasible_reason} Please consider refine the plan."
                break
        
        return None

    def final_output(self):
        """
        Just output the root node as the terminal node, no need of children
        """
        self.SubTask.stop_flag = True

        if self.SubTask.complete_flag:
            # self.env stay at the final state
            output_node = self.final_terminal_node
            self.WebTask.final_answer = self.SubTask.final_answer # set the final answer of the WebTask
            self.SubTask.executed_actions = self.executed_scroll_actions
            logging_finishing_subtask_info(output_node, self.SubTask.final_answer, self.env.page.url, self.actree, [])

            # update the WebTask
            self.WebTask.finished_subtasks.append(self.SubTask)
            self.WebTask.step_sum += len(self.SubTask.executed_actions)
            self.WebTask.executed_actions.extend(self.SubTask.executed_actions)

        else:
            output_node = self.root
            self._env_goto_node(output_node)
            self.obs = self.info_for_trajectory["observation"]

        self.obs["text"] = trimm_actree(self.obs["text"])
        transfer_screenshot(self.root, self.WebTask.task_id, self.SubTask.idx+1)

        return self.env, self.obs, self.WebTask, self.SubTask, output_node,



# ==================== transfer screenshot ====================
def transfer_screenshot(node, task_id, subtask_idx):
    try:
        if not os.path.exists(f"task_info/{task_id}/node_pics"):
            os.makedirs(f"task_info/{task_id}/node_pics")
        shutil.copy(f"webpilot_utils/node_pics/{node.name}.pdf", f"task_info/{task_id}/node_pics/{node.name}.pdf")
    except:
        pass

    try:
        if not os.path.exists(f"task_info/{task_id}/gvs"):
            os.makedirs(f"task_info/{task_id}/gvs")
        shutil.copy(f"{task_id}_{subtask_idx}.gv",
                    f"task_info/{task_id}/gvs/{task_id}_{subtask_idx}.gv")
        shutil.copy(f"{task_id}_{subtask_idx}.gv.pdf",
                    f"task_info/{task_id}/{task_id}_{subtask_idx}.pdf")
    except:
        pass

    return None