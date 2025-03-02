

from WebPilot.env_utils import *
from WebPilot.prompter.executor import *
from WebPilot.agent_func.controller import is_subtask_stopped
from WebPilot.agent_func.verifier import ActionVerifier

# ==================== single action ====================
# singel action_generation. including possible stop action
def gen_next_action(actree, WebTask, SubTask, parent_node_sim_reflection, parent_node_reflection_for_child, parent_not_stop_reason="", parent_action=None):
     """
     used when expanding to a brand new node
     mostly in expansion part when new node is generated
     """

     # last time subtask execution reflection only used for the first layer node
     last_subtask_reflection = SubTask.last_subtask_reflection

     finished_subtasks_str = ""
     for idx, subtask in enumerate(WebTask.finished_subtasks):
          finished_subtasks_str += f"{idx+1}. {subtask.content}\n"

     prompt = gen_next_action_prompt(WebTask=WebTask, SubTask=SubTask,
                                     finished_subtasks=finished_subtasks_str,
                                     task_content=SubTask.content,
                                     domain=WebTask.domain,
                                     actree=actree,
                                     executed_actions=gen_readable_actions(SubTask.executed_actions),
                                     parent_node_sim_reflection=parent_node_sim_reflection,
                                     parent_node_reflection_for_child=parent_node_reflection_for_child,
                                     not_stop_reason=parent_not_stop_reason,
                                     last_subtask_reflection=last_subtask_reflection,
                                     )
     response = ask_LLM(prompt)
                         
     # generate action-element pair from the output of LLM
     action_generation_dict = json.loads(response)
     
     # regularize the action
     action_dict = ActionVerifier.format_regularizer(actree, action_generation_dict, prompt)
     
     # add element_info: dom_info which is the line in the actree
     action_dict = add_element_info(action_dict, actree)
     
     # deduplicate the action w.r.t its parent
     action_dict_deduplicated = ActionVerifier.action_deduplicator(actree, action_dict, prompt, [parent_action])
     if isinstance(action_dict_deduplicated, str):
          pass
     else:
          action_dict = action_dict_deduplicated

     # add the prompt used for action generation to the action_dict
     action_dict["gen_action_prompt"] = prompt
     
     return action_dict

# ==================== generate single action with siblings reflection ====================
def gen_action_with_reflection(actree, WebTask, SubTask, sibling_actions:list, sibling_actions_n_reflections:list[str], 
                               parent_node_sim_reflection, parent_node_reflection_for_child,
                               parent_not_stop_reason="",
                               parent_action=None):
     """
     Works for expansion empty node, when at least one sibling visited, asking for action with siblings reflection
     """

     last_subtask_reflection = SubTask.last_subtask_reflection

     # construct the finished subtasks str
     finished_subtasks_str = ""
     for idx, subtask in enumerate(WebTask.finished_subtasks):
          finished_subtasks_str += f"{idx+1}. {subtask.content}\n"
     
     prompt = gen_next_action_with_reflection_prompt(WebTask=WebTask, SubTask=SubTask,
                                                       finished_subtasks=finished_subtasks_str,
                                                       task_content=SubTask.content,
                                                       domain=WebTask.domain,
                                                       actree=actree,
                                                       executed_actions=gen_readable_actions(SubTask.executed_actions),
                                                       sibling_actions_n_reflections=sibling_actions_n_reflections,
                                                       parent_node_sim_reflection=parent_node_sim_reflection,
                                                       parent_node_reflection_for_child=parent_node_reflection_for_child                                                 ,
                                                       not_stop_reason=parent_not_stop_reason,
                                                       last_subtask_reflection=last_subtask_reflection,
                                                       )
     response = ask_LLM(prompt)
     action_generation_dict = json.loads(response)
     action_dict = ActionVerifier.format_regularizer(actree, action_generation_dict, prompt)
     action_dict = add_element_info(action_dict, actree)
     action_dict["gen_action_prompt"] = prompt

     # add action_deduplicator to avoid the same action as its siblings, as well as its parent
     action_dict = ActionVerifier.action_deduplicator(actree, action_dict, prompt, sibling_actions+[parent_action])
     
     return action_dict

# ==================== generate single action with stop asking====================
def gen_action_with_stop_asking(actree, WebTask, SubTask, node):
     """
     used in simulation part, only generate one action.
     """

     # stop asking
     subtask_stop_reason, is_stopped, _, _ = \
          is_subtask_stopped(actree, WebTask, SubTask, node.node_reflection_for_child, node.observation_description)
     if is_stopped:
          if SubTask.need_answer:
               prompt = gen_subtask_final_answer_prompt(task_content=SubTask.content, 
                                                       domain=WebTask.domain,
                                                       expectation=SubTask.expectation, 
                                                       actree=actree, 
                                                       observation_description=node.observation_description)
               response = ask_LLM(prompt)
               intermediate_info = json.loads(response)["final_answer"]
          else:
               intermediate_info = ""

          return subtask_stop_reason, {"action_type": "NONE", "action_str": "stop", "action_intent": None, "element_info": None}, intermediate_info
     else:
          not_stop_reason = subtask_stop_reason

     # get next action
     if is_stopped == False:
          logging.info("===== Current process: is_subtask_stopped==False, so generate a new action for simulation =====")
     elif is_stopped == True:
          logging.info("===== Current process: is_subtask_stopped==True, but the subtask_verifier rectified it, so generate a new action for simulation =====")
     logging.info(f"===== Current Node name: {node.name[:4]} =====")
     logging.info(f"===== Current Node url: {node.url} =====")

     # construct the finished subtasks str
     finished_subtasks_str = ""
     for idx, subtask in enumerate(WebTask.finished_subtasks):
          finished_subtasks_str += f"{idx+1}. {subtask.content}\n"
     
     prompt = gen_next_action_prompt(WebTask=WebTask, SubTask=SubTask,
                                     finished_subtasks=finished_subtasks_str,    
                                     task_content=SubTask.content,
                                     domain=WebTask.domain,
                                     actree=actree,
                                     executed_actions=gen_readable_actions(SubTask.executed_actions),
                                     parent_node_sim_reflection="",
                                     parent_node_reflection_for_child=node.node_reflection_for_child,
                                     not_stop_reason=not_stop_reason,
                                     )
     response = ask_LLM(prompt)

     # generate action-element pair from the output
     action_generation_dict = json.loads(response)

     # regularize the action
     action_dict = ActionVerifier.format_regularizer(actree, action_generation_dict, prompt)

     # add element_info: dom_info which is the line in the actree
     action_dict = add_element_info(action_dict, actree)
     # deduplicate the action w.r.t its parent
     action_dict_deduplicated = ActionVerifier.action_deduplicator(actree, action_dict, prompt, [node.src_action])
     if isinstance(action_dict_deduplicated, str): 
          pass
     else:
          action_dict = action_dict_deduplicated

     # 7. add the prompt used for action generation to the action_dict
     action_dict["gen_action_prompt"] = prompt

     # also output the reason(even not stop) as the new ScrathPad_thoughts
     return not_stop_reason, action_dict, ""


# ==================== re-generate action for interact_verifier ====================
def re_gen_action(former_gen_action_prompt, failed_execution_reflection, failed_action, actree):
    """
    in interact_verifier, re-generate the action for failed action
    """

    # simple action representation
    simple_failed_action = {"action_type": failed_action["action_type"],
                            "action_str": failed_action["action_str"],
                            "element_info": failed_action["element_info"],
                            "action_intent": failed_action["action_intent"],}

    prompt = gen_re_gen_action_prompt(former_gen_action_prompt=former_gen_action_prompt,
                                    failed_execution_reflection=failed_execution_reflection,
                                    failed_action=repr(simple_failed_action),
                                    )
    response = ask_LLM(prompt)

    action_generation_dict = json.loads(response)
    action_dict = ActionVerifier.format_regularizer(actree, action_generation_dict, prompt)
    action_dict = add_element_info(action_dict, actree)
    action_dict["gen_action_prompt"] = prompt

    return action_dict

# ==================== executable action for webarena ====================
def gen_executable_action(action_dict):
     """
     genereate the executable action for the webarena
     """
     action_type = action_dict["action_type"]
     if action_type == "noop":
          return create_none_action()

     # if action_str is list, form the elements into the format of "[] [] []", means "type"
     if isinstance(action_dict["action_str"], list):
          action_str = ""
          for element in action_dict["action_str"]:
               action_str += f"[{element}] "
          executable_action = create_id_based_action(f"{action_type} {action_str}")

     else:
          action_str = action_dict["action_str"]
          executable_action = create_id_based_action(f"{action_type} [{action_str}]")
     
     return executable_action

# ==================== empty text before type ====================
def empty_text_before_type(type_action, env) -> dict:
     """
     Not actually used, but changed the source code of webarena's env.
     Args:
         type_action (_type_): src_action dict
         env (_type_): playwrigth env

     Returns:
         dict: action_dict, transfer the id for current actree
     """

     logging.info("===== Empty text before type =====")

     # 0. get the element id from the type_action
     element_id = type_action["element_id"]

     # 1. focus on the textbox
     focus_textbox_action = create_id_based_action(f"click [{element_id}]")
     env.step(focus_textbox_action)

     # 3. select all 
     if 'mac' in env.page.evaluate("navigator.platform").lower():
          key_comb = "Meta+A"
     else:
          key_comb = "Control+A"
     select_all_action = create_id_based_action(f"press [{key_comb}]")
     env.step(select_all_action)

     # 3. using delete
     delete_action = create_id_based_action(f"press [Backspace]")
     env.step(delete_action)

     return None


# ==================== judge whether the base of url is the same ====================
def is_same_base_url(url1:str, url2:str) -> bool:
     """
     judge whether the base of url is the same
     """
     base_url1 = urlparse(url1).netloc + urlparse(url1).path
     base_url2 = urlparse(url2).netloc + urlparse(url2).path

     return base_url1 == base_url2


# ==================== node reflection when entering a new node after obs_des and before evaluation ====================
def node_reflection(actree, WebTask, SubTask, src_action, obs_des_dict):
     """
     This works for real child node, combine the observation_description and src_action to get a node_reflection
     """

     src_action_str = gen_readable_actions([src_action], with_intent=False, with_effect=False)
     src_action_with_intent = gen_readable_actions([src_action], with_intent=True, with_effect=False)
     node_reflection_dict = {}
     
     # for child
     obs_des_changes = f"**Description**: {obs_des_dict['description']}\n" \
          + f"**Changes**: {obs_des_dict['changes']}\n" 
     executed_action_str = gen_readable_actions(SubTask.executed_actions, with_intent=False, with_effect=False,
                                                   with_intent_fulfillment=True)
     prompt = gen_node_reflection_prompt.for_child(task_content=SubTask.content, domain=WebTask.domain, expectation=SubTask.expectation,
                                                   actree=actree, 
                                                   executed_actions=executed_action_str,
                                                   obs_des_changes=obs_des_changes,)
     response = ask_LLM(prompt)
     node_reflection_dict.update(json.loads(response))

     # for sibling
     obs_des_fulfillment = f"**Description**: {obs_des_dict['description']}\n"
     action_intent_fulfillment = f"**Action Intent Fulfillment**: {obs_des_dict['action_intent_fulfillment']}\n"
     obs_des_fulfillment += action_intent_fulfillment

     prompt = gen_node_reflection_prompt.for_sibling(task_content=SubTask.content, domain=WebTask.domain, expectation=SubTask.expectation,
                                                     actree=actree,
                                                     executed_actions=gen_readable_actions(SubTask.executed_actions[:-1], with_intent=False, with_effect=False),
                                                     src_action_with_intent=src_action_with_intent,
                                                     obs_des_fulfillment=obs_des_fulfillment,)
     response = ask_LLM(prompt)
     node_reflection_dict.update(json.loads(response))

     node_reflection_for_child = node_reflection_dict["node_reflection_for_child"]
     node_reflection_for_sib = node_reflection_dict["node_reflection_for_sib"]

     # add the action str to the node_reflection
     node_reflection_for_child = f"- {src_action_str}- Reflection: {node_reflection_for_child}"
     node_reflection_for_sib = f"- {src_action_str}- Reflection: {node_reflection_for_sib}"
     
     return node_reflection_for_child, node_reflection_for_sib

# ==================== simulation reflection for real child node ====================
def sim_reflection(actree, SubTask, WebTask, src_action, sim_obs_des_dict, score, score_reason):
     """
     This works for simulated node, combine the observation_description and score_reason to get a sim_reflection.
     The generated sim_reflection will be used for its parent's real children nodes.
     """

     task_content = SubTask.content

     obs_des_changes_fulfillment = f"**Description**: {sim_obs_des_dict['description']}\n" \
          + f"**Changes**: {sim_obs_des_dict['changes']}\n" \
          + f"**Action Intent Fulfillment**: {sim_obs_des_dict['action_intent_fulfillment']}"

     prompt = gen_sim_reflection_prompt(task_content=task_content,
                                        expectation=SubTask.expectation,
                                        domain=WebTask.domain,
                                        actree=actree,
                                        src_action=gen_readable_actions([src_action], with_intent=True, with_effect=False),
                                        ScratchPad_Info="",
                                        obs_des_changes_fulfillment=obs_des_changes_fulfillment,
                                        score_reason=score_reason,
                                        )
     response = ask_LLM(prompt)
     sim_reflection = json.loads(response)["sim_reflection"]

     src_action_str = f"""[action_type]: {src_action["action_type"]}, [element_info]: {src_action["element_info"]}. [action_intent]: {src_action["action_intent"]}."""

     # construct the sim_reflection with more structured info
     reflection = f"""I am doing an MCTS searching, I have simulated the action: {src_action_str}
My reflection on it is: {sim_reflection}

"""
     return reflection

# ==================== terminal reflection ====================
def terminal_reflection(actree, WebTask, SubTask, former_node_reflections, score_reason):
    """
    This function is used for reflection, if a node is terminal, then we ask agent to reflect itself to summarize the reason of terminal(maybe success or failure)
    Could be used for the whole search process, since the max_terminal_nodes is limited.
    """

    task_content = SubTask.content
    ScratchPad_Info = ""
    prompt = gen_terminal_reflection_prompt(task_content=task_content,
                                            domain=WebTask.domain,
                                            actree=actree,
                                            executed_actions=gen_readable_actions(SubTask.executed_actions),
                                            ScratchPad_Info=ScratchPad_Info,
                                            former_node_reflections=former_node_reflections,
                                            score_reason=score_reason,
                                            )
    response = ask_LLM(prompt)
    reflection = json.loads(response)["terminal_reflection"]

    return reflection
