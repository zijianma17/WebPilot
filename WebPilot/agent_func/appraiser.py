

from WebPilot.env_utils import *
from WebPilot.prompter.appraiser import *


# ==================== state evaluation ====================
def node_evaluation(actree, WebTask, SubTask, executed_actions, obs_des_dict,):
     """
     evaluate the node, return the reason and score. reason for analysis, also helpful for reasonable score
     """
     
     if "necessary actions" in SubTask.expectation.lower():
          executed_action_str = gen_readable_actions(executed_actions, with_intent=False, with_effect=False)
     else:
          executed_action_str = gen_readable_actions([executed_actions[-1]], with_intent=False)

     obs_des_n_fulfillment = f"**Description**: {obs_des_dict['description']}\n" \
          + f"**Action Intent Fulfillment**: {obs_des_dict['action_intent_fulfillment']}"

     prompt = gen_node_evaluation_prompt(task_content=SubTask.content,
                                             domain=WebTask.domain,
                                             actree=actree,
                                             executed_actions=executed_action_str,
                                             ScratchPad_Info="",
                                             obs_des_n_fulfillment=obs_des_n_fulfillment,
                                             expectation=SubTask.expectation,
                                             )
     response = ask_LLM(prompt)
     node_evaluation_dict = json.loads(response)
     reasoning_process = node_evaluation_dict["reasoning_process"]
     executed_action_score = float(node_evaluation_dict["executed_action_score"])
     future_promise_score = float(node_evaluation_dict["future_promise_score"])
     score = 0.5 * executed_action_score + 0.5 * future_promise_score
     score_list = [score, executed_action_score, future_promise_score]

     return reasoning_process, score_list

# ==================== state evaluation for terminal nodes ====================
def terminal_evaluation(actree, observation_description, WebTask, SubTask,):
     """
     The info which is used for action_generation will be used again, for reflection. Then get a new reward.
     """

     task_content = SubTask.content

     prompt = gen_terminal_evaluation_prompt(task_content=task_content,
                                             domain=WebTask.domain,
                                             expectation=SubTask.expectation,
                                             actree=actree,
                                             observation_description=observation_description,
                                             executed_actions=gen_readable_actions(SubTask.executed_actions),
                                             )
     response = ask_LLM(prompt)
     terminal_evaluation_dict = json.loads(response)

     # return plan_score, execution_score, state_score, score_reason
     reasoning_process = terminal_evaluation_dict["reasoning_process"]
     score = float(terminal_evaluation_dict["score"])

     return reasoning_process, score

# ==================== terminal comparison for more than one terminal nodes ====================
def terminal_comparison(nodes_list, WebTask, SubTask, executed_actions_list):
     """
     compare the terminal nodes, and return the best one (better one)
     currently only compare two nodes
     """
     task_content = SubTask.content
     node_info_1 = [nodes_list[0].actree, nodes_list[0].observation_description]
     node_info_1.append(gen_readable_actions(executed_actions_list[0]))
     node_info_2 = [nodes_list[1].actree, nodes_list[1].observation_description]
     node_info_2.append(gen_readable_actions(executed_actions_list[1]))

     prompt = gen_terminal_comparison_prompt(task_content=task_content,
                                             domain=WebTask.domain,
                                             expectation=SubTask.expectation,
                                             node_info_1=node_info_1,
                                             node_info_2=node_info_2,
                                             )
     response = ask_LLM(prompt)
     comparison_dict = json.loads(response)

     # convert the possible string to int
     best_node_idx = int(comparison_dict["better_node_idx"])

     return best_node_idx
