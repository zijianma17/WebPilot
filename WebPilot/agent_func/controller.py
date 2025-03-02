
from WebPilot.env_utils import *
from WebPilot.prompter.controller import *
from WebPilot.model import (
    MAJORITY_VOTE_FOR_STOP_FLAG,
    SKIP_COMPLETE_ESTIMATION_FLAG,
)

# ==================== is_subtask_stopped ====================
def is_subtask_stopped(actree, WebTask, SubTask, node_reflection, observation_description):
     """
     judge whether the subtask is stopped, the env may not be terminated if the subtask is not the last one
     """

     is_subtask_stopped_dict = {}
     if not "**Necessary Actions**" in SubTask.expectation: 
          # observation
          prompt = gen_is_subtask_stopped_prompt.observation(WebTask.domain, SubTask.expectation_observation, actree, observation_description)
          response = ask_LLM(prompt)
          is_subtask_stopped_dict.update(json.loads(response))
          # no necessary actions needed
          logging.info("No necessary actions needed for this subtask. Directly set necessary_actions_executed to True.")
          is_subtask_stopped_dict["necessary_actions_executed"] = True
          is_subtask_stopped_dict["reasoning_of_actions"] = ""
     
     else:
          # action
          prompt = gen_is_subtask_stopped_prompt.actions(WebTask.domain, SubTask.expectation_action, 
                                                         gen_readable_actions(SubTask.executed_actions, with_intent=False, with_effect=False))
          response = ask_LLM(prompt)
          is_subtask_stopped_dict.update(json.loads(response))
          # no observation needed
          logging.info("Focusing on the necessary actions. No need to analyze the observation. Set observation_meets_criteria to True.")
          is_subtask_stopped_dict["observation_meets_criteria"] = True
          is_subtask_stopped_dict["reasoning_of_observation"] = ""

     # reflection
     prompt = gen_is_subtask_stopped_prompt.reflection(WebTask.domain, SubTask.content, SubTask.expectation, node_reflection)
     response = ask_LLM(prompt)
     is_subtask_stopped_dict.update(json.loads(response))

     # gather the flags
     is_s_s_keys = [str2bool(is_subtask_stopped_dict["observation_meets_criteria"]),
                    str2bool(is_subtask_stopped_dict["necessary_actions_executed"]),
                    not str2bool(is_subtask_stopped_dict["reflection_suggests_further_actions"])] 
     
     if "Necessary Actions" in SubTask.expectation: # for subtask with necessary actions, don't let the reflection decide
          is_subtask_stopped_dict["reflection_suggests_further_actions"] = False 
     
     # logging the results for comparsion
     logging.info("is_subtask_stopped finished, the results are as follows:")
     # construct the log_str
     log_str = f"\nObservation meets criteria: {is_s_s_keys[0]}"
     log_str += f"\nNecessary actions executed: {is_s_s_keys[1]}"
     log_str += f"\n(Not) Reflection suggests further actions: {is_s_s_keys[2]}"
     log_str += f"\nSo the final decision is: {is_s_s_keys[0] and is_s_s_keys[1] and is_s_s_keys[2]}\n\n\n"
     is_s_s_str = log_str
     logging.info(is_s_s_str)

     if not MAJORITY_VOTE_FOR_STOP_FLAG:
          reason = is_subtask_stopped_dict["reason"]
          subtask_completeness = is_subtask_stopped_dict["subtask_completeness"]
          # the answer could be bool or str of (True/False)
          stop_decision = str2bool(is_subtask_stopped_dict["stop_decision"])
     else:
          # reason = is_subtask_stopped_dict["reason"]
          reason = "" #

          # majority vote for stop
          stop_1 = str2bool(is_subtask_stopped_dict.get("observation_meets_criteria", False))
          stop_2 = str2bool(is_subtask_stopped_dict.get("necessary_actions_executed", False))
          stop_3 = not str2bool(is_subtask_stopped_dict.get("reflection_suggests_further_actions", True))
          
          if stop_1 + stop_2 + stop_3 >= 3:
               stop_decision = True
          else:
               stop_decision = False
               # construct a new reason for not stopping, specifically for the aspect with "not stop"
               reason = ""
               if stop_1 == False:
                    reason += is_subtask_stopped_dict["reasoning_of_observation"] + "\n"
               if stop_2 == False:
                    reason += repr(is_subtask_stopped_dict["reasoning_of_actions"]) + "\n"
               if stop_3 == False:
                    pass

     subtask_completeness = ""
     return reason, stop_decision, subtask_completeness, is_s_s_str

# ==================== subtask stop verifier ====================
def subtask_stop_verifier(WebTask, SubTask, actree, is_s_s_log_str):
     """
     Actually like ask one more time about whether the subtask is stopped.
     But the input will be more subjective, and reflection are removed to avoid misleading.
     Only be called after the is_subtask_stopped, to verify the result.
     """
     single_obs_des = gen_single_observation_description(SubTask.content, actree)
     SubTask.detail_observation_description = f"""**Description**: {single_obs_des['overall_description']} {single_obs_des['main_body']}\n*Task Specific Elements Status**: {single_obs_des['task_specific_elements_status']}"""

     # observation
     prompt = gen_subtask_stop_verifier_prompt.observation(WebTask.domain, actree, 
                                                           SubTask.detail_observation_description, 
                                                           SubTask.expectation)
     response = ask_LLM(prompt)
     verifier_dict = json.loads(response)
     # actions
     prompt = gen_subtask_stop_verifier_prompt.actions(WebTask.domain, SubTask.expectation, 
                                                       gen_readable_actions(SubTask.executed_actions))

     response = ask_LLM(prompt)
     verifier_dict.update(json.loads(response))

     # gather the flags
     s_s_v_keys = [str2bool(verifier_dict["observation_meets_criteria"]),
                    str2bool(verifier_dict["necessary_actions_executed"]),]                          
     
     # logging the results for comparsion
     logging.info("previously is_subtask_stopped results:")
     logging.info(is_s_s_log_str)
     logging.info("subtask_stop_verifier finished, the results are as follows:")
     # construct the log_str
     log_str = f"\nObservation meets criteria: {s_s_v_keys[0]}"
     log_str += f"\nNecessary actions executed: {s_s_v_keys[1]}"
     log_str += f"\nSo the final decision is: {s_s_v_keys[0] and s_s_v_keys[1]}\n\n\n"
     s_s_v_str = log_str
     logging.info(s_s_v_str)

     if not MAJORITY_VOTE_FOR_STOP_FLAG:
          reason = verifier_dict["reason"]
          stop_decision = str2bool(verifier_dict["stop_decision"])
     else:
          # reason = verifier_dict["reason"]
          reason = ""
          stop_1 = str2bool(verifier_dict.get("observation_meets_criteria", False))
          stop_2 = str2bool(verifier_dict.get("necessary_actions_executed", False))

          if stop_1 + stop_2 >= 2:
               stop_decision = True
          else:
               stop_decision = False
               reason = ""
               if stop_1 == False:
                    reason += verifier_dict["reasoning_of_observation"] + "\n"
               if stop_2 == False:
                    reason += verifier_dict["reasoning_of_actions"] + "\n"

     subtask_completeness = ""
     complete_flag = True

     return reason, stop_decision, subtask_completeness, complete_flag

# ==================== subtask_completeness_estimator (for force-stopped terminal node) ====================
def subtask_completeness_estimator(WebTask, SubTask, root_actree, actree, reflections_for_sib, forced_terminal_flag=False):
     """
     The subtask is stopped, then we need to estimate the completion of the subtask.
     """

     if SKIP_COMPLETE_ESTIMATION_FLAG:
          return "", True, ""
               
     # if Subtask already has the detail_observation_description, then use it, otherwise generate it.
     if not hasattr(SubTask, "detail_observation_description"):
          single_obs_des = gen_single_observation_description(SubTask.content, actree)
          SubTask.detail_observation_description = f"""**Description**: {single_obs_des['overall_description']} {single_obs_des['main_body']}\n"""

     completion_dict = {}
     if not "**Necessary Actions**" in SubTask.expectation:
          # observation
          prompt = gen_subtask_completeness_estimator_prompt.observation(SubTask.content, WebTask.domain, SubTask.expectation_observation,
                                                                      actree, SubTask.detail_observation_description)
          response = ask_LLM(prompt)
          completion_dict.update(json.loads(response))
          subtask_completeness = completion_dict["observation_analysis"] + " " + completion_dict["task_completeness"]

     else:
          # action
          prompt = gen_subtask_completeness_estimator_prompt.actions(SubTask.content, WebTask.domain, SubTask.expectation_action,
                                                                      exeucted_actions=gen_readable_actions(SubTask.executed_actions, with_intent=False, with_effect=False),
                                                                      actions_n_refs_for_sib=reflections_for_sib,
                                                                      )
          response = ask_LLM(prompt)
          completion_dict.update(json.loads(response))
          subtask_completeness = completion_dict["task_completeness"]

     complete_flag = str2bool(completion_dict["complete_flag"])

     if complete_flag:
          subtask_reflection = ""
     else:               
          # sub_reflection
          if not "**Necessary Actions**" in SubTask.expectation:
               observation_analysis = subtask_completeness
               action_analysis = ""
               executed_actions_str = gen_readable_actions(SubTask.executed_actions, with_intent=False, with_effect=True)
          else:
               observation_analysis = ""
               action_analysis = subtask_completeness
               executed_actions_str = gen_readable_actions(SubTask.executed_actions, with_intent=False, with_effect=False)

          prompt = gen_subtask_completeness_estimator_prompt.sub_reflection(SubTask.content, WebTask.domain, SubTask.expectation,
                                                                           root_actree, observation_analysis, action_analysis,
                                                                           executed_actions=executed_actions_str,
                                                                           )
          response = ask_LLM(prompt)
          completion_dict.update(json.loads(response))
          subtask_reflection = completion_dict["task_reflection"]                                                                       

     return subtask_completeness, complete_flag, subtask_reflection
