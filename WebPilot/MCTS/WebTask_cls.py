from typing import Any
import re
import logging
import json
import shutil

from WebPilot.model import (
     DECOMPOSE_FLAG,
     MAJORITY_VOTE_FOR_STOP_FLAG,
)

from WebPilot.model import ask_LLM

# prompts
from WebPilot.prompter.planner import *
from WebPilot.prompter.controller import *
from WebPilot.prompter.extractor import *
from WebPilot.prompter.executor import *
from WebPilot.prompter.appraiser import *
from WebPilot.prompter.verifier import *


from WebPilot.env_utils import *
from WebPilot.agent_func.all_agent_funcs import *

from .SubTask_cls import SubTask_cls
from .MCTS_cls import rename_incomplete_subtasks_files


# ==================== WebTask_cls ====================


class WebTask_cls:
     def __init__(self, task_content, task_id, config_file, actree):
          self.content = task_content
          self.task_id = task_id
          self.config_file = config_file
          self.subtasks = []
          self.finished_subtasks = []
          self.step_sum = 0
          self.executed_actions = []
          self.trajectory = []
          self.stop_flag = False
          self.need_answer = False
          self.final_answer = ""

          self.which_domain()

          self.task_expectation()

          if "**Answer Requirements**" in self.expectation:
               self.need_answer = True
               self.transform_answer_info()

          self.whether_decompose(actree) # generate the plan

     def which_domain(self):
          """
          will be excuted when WebTask is initialized
          """
          config_info = json.load(open(self.config_file, "r"))
          self.domain = config_info["sites"][0]
          if self.domain == "wikipedia":
               self.domain = "map"
          logging.info("Domain of the task is: " + self.domain)
          return None

     def task_expectation(self):
          """
          use the golden_expectation as examples to generate the task expectation
          """
          expectation_examples = ""
          prompt = gen_task_expectation_prompt(domain=self.domain,
                                                  task_content=self.content,
                                                  examples=expectation_examples)
          response = ask_LLM(prompt)
          expectation_dict = json.loads(response)
          self.expectation_observation = f"**Target Page Description**: {expectation_dict['target_page_description']}\n"
          self.expectation = self.expectation_observation
          if str2bool(expectation_dict["need_answer"]):
               self.expectation_answer = f"**Answer Requirements**: {expectation_dict['answer_requirements']}\n"
               self.expectation += self.expectation_answer
          else:
               self.final_answer = "N/A"

          return None

     def is_answer_needed(self):
          """
          let the agent decie whether the final answer is needed.
          add the response to the Expectation.
          """
          prompt = gen_is_answer_needed_prompt(
               self.content, self.domain, self.expectation,)
          response = ask_LLM(prompt)
          self.need_answer = str2bool(json.loads(response)["need_answer"])
          if self.need_answer:
               self.expectation_answer = "\n**Answer Requirements**: " + json.loads(response)["answer_requirements"]
               self.expectation += "\n" + self.expectation_answer

          return None
     
     def transform_answer_info(self):
          """
          transform the answer requirements into a phrase
          """
          prompt = gen_answer_description_prompt(self.expectation_answer, self.domain)
          response = ask_LLM(prompt)
          answer_dict = json.loads(response)
          self.info_extraction_dict = {
               "info_needed": answer_dict["info_needed"],
               "info_requirements": answer_dict["info_requirements"],}
          
          # generate the subtask_dict for the info_extraction subtask
          info_to_find = self.content

          self.info_extraction_subtask_dict = {
               "subtask": "Find the info: " + info_to_find,
               "expectation": "**Answer Requirements**: " + self.info_extraction_dict.get("info_requirements", "") + "\n",
               "info_extraction_flag": True,
          }

          return None

     def whether_decompose(self, actree):
          """
          let the agent think if the current task is too complex and needs to be decomposed.
          here if the answer is True, also generate the whole plan: first rough plan.
          """
          observation_description = gen_single_observation_description(
               self.content, actree)
          observation_description = observation_description["overall_description"] + \
               observation_description["main_body"]

          if DECOMPOSE_FLAG:
               self.decompose_flag = True
          prompt = gen_generate_plan_prompt(task_domain=self.domain,
                                             task_content=self.content,
                                             task_expectation=self.expectation,
                                             observation_description=observation_description,
                                             )          
          response = ask_LLM(prompt,)

          plan_dict = json.loads(response)
          self.plan = plan_dict["plan"]

          for subtask in self.plan:
               if isinstance(subtask.get("necessary_actions", ""), list):
                    n_a_str = "The necessary actions should include: "
                    n_a_str += ", ".join(subtask["necessary_actions"])
                    subtask["necessary_actions"] = n_a_str

          # construct the plan_str
          plan_str = "========== Current Task ==========\n"
          plan_str += f"{self.task_id}. {self.content}\n"
          plan_str += "========== Current Plan ==========\n"
          for idx, subtask_dict in enumerate(self.plan):
               plan_str += str(idx+1) + ". \n"
               for key, value in subtask_dict.items():
                    plan_str += f"\t**{key}**: {value}\n"
               
          plan_str += "===================================\n\n\n"
          print_n_log(plan_str)
          print_n_log("Plan generated successfully. The final plan is as above.")

          return None

     def gen_subtask_from_plan_first(self, actree):
          """
          After the plan is updated, get the first subtask from the plan
          """

          # judge whether the first subtask is info_extraction
          if str2bool(self.plan[0].get("need_answer", False)):
               # final_subtask = self.plan.pop(0)
               self.plan.insert(0, self.info_extraction_subtask_dict)

          # get the first subtask from the plan
          subtask_content = self.plan[0]["subtask"]
          SubTask = SubTask_cls(content=subtask_content, idx=len(self.subtasks))
          target_page_description = self.plan[0].get("target_page_description", "")
          necessary_actions = self.plan[0].get("necessary_actions", "")

          SubTask.expectation_observation = f"**Target Page Description**: {target_page_description}\n"
          SubTask.expectation = SubTask.expectation_observation

          if necessary_actions.strip() != "" and not ("not needed" in necessary_actions.lower()): # means necessary_actions is a must
               SubTask.expectation_action = f"**Necessary Actions**: {necessary_actions}\n"
               SubTask.expectation += SubTask.expectation_action
          else:
               SubTask.expectation_action = ""

          # add last_subtask_reflection to the subtask
          SubTask.last_subtask_reflection = self.plan[0].get("last_subtask_reflection", "")

          if self.need_answer and "info_extraction_flag" in self.plan[0].keys():
               SubTask.need_answer = True
               # SubTask.gen_answer_requirements(domain=self.domain, actree=actree) # requirements already in subtask.attributes
               SubTask.interaction_type = "info_extraction"
               SubTask.expectation = self.plan[0].get("expectation")
          else:
               SubTask.need_answer = False
               SubTask.interaction_type = "web_interaction"

          self.subtasks.append(SubTask)

          # show the subtask generated
          print_n_log(
               f"Subtask generated: Index: {SubTask.idx+1}, Content: {SubTask.content}")
          print_n_log(f"Subtask expectation: {SubTask.expectation}")
          print_n_log(f"Subtask interaction type: {SubTask.interaction_type}")

          return SubTask
     
     def whether_stop_update_plan(self, final_actree, final_subtask,):
          """
          last subtask is finished, judge whether stop, and then judge the next subtask, whether it should be the real next one to execute;
          """
          
          subtask_complete_flag = final_subtask.complete_flag

          finished_subtasks_str = ""
          for subtask in self.finished_subtasks:
               finished_subtasks_str += f"""{subtask.idx+1}. **Subtask**: {subtask.content}\n**Subtask Completeness**: {subtask.subtask_completeness}\n"""

          plan_str = ""  # construct a plan string without expectation
          # if the subtask is completed, then start from the second subtask
          plan_list = self.plan[1:] if subtask_complete_flag else self.plan
          for idx, subtask_dict in enumerate(plan_list):
               plan_str += str(idx+1) + ". " + subtask_dict["subtask"] + "\n"

          if self.finished_subtasks != [] and self.finished_subtasks[-1].interaction_type == "info_extraction":
               last_subtask_is_infoextraction = True
          else:
               last_subtask_is_infoextraction = False

          if subtask_complete_flag == False: # re-do the current subtask
               logging.info(
                    "===== The last subtask is incomplete, skip the webtask_stop_asking part and redo the current subtask. =====")
               rename_incomplete_subtasks_files(self.task_id, len(self.subtasks))
               self.subtasks.pop() # plan remains unchanged
               
               subtask_reflection = final_subtask.subtask_reflection

               self.plan[0]["last_subtask_reflection"] = subtask_reflection

               return None
 
          # ========== following is the normal process (last subtask is completed) ==========
          self.plan.pop(0) # pop the first subtask in the plan, since it's completed
          if self.plan == [] and not self.need_answer:
               # need_answer
               logging.info("===== Current process: The plan is empty, stop the webtask. =====")
               self.stop_flag = True
               return None

          # if need answer, then judge whether call info_extractor
          if self.need_answer and not last_subtask_is_infoextraction:
               if self.plan == []: # insert an info_extractor
                    logging.info("===== Current process: The plan is empty, but need answer, generate a info_extraction subtask. =====")
                    is_info_extraction = True
               elif str2bool(self.plan[0]["need_answer"]):
                    is_info_extraction = True
               else:
                    is_info_extraction = False               

               if is_info_extraction:
                    # delete the first subtask in the current plan, and insert the info_extraction subtask
                    self.plan.pop(0) if self.plan != [] else None # o
                    self.plan.insert(0, self.info_extraction_subtask_dict)
                    return None
               else:
                    # not_stop_reason += "\n" + not_info_extraction_reason
                    pass
          elif self.need_answer and last_subtask_is_infoextraction:
               logging.info("===== Current process: The last subtask is info_extraction, so the info is already extracted. Stop the webtask. =====")
               self.stop_flag = True
               return None
          elif not self.need_answer:
               logging.info("===== Current process: The webtask doesn't need answer, skip the whether_call_info_extractor part. =====")

          
          # update the plan
          self.next_subtask_to_execute()
          
          # after update decision, if the plan is empty, then stop the webtask
          if self.plan == []:
               if not self.need_answer:
                    logging.info("===== Current process: The plan is empty, and doesn't need answer, stop the webtask. =====")
                    self.stop_flag = True
               else:
                    logging.info("===== Current process: The plan is empty, but need answer, generate a info_extraction subtask. =====")
                    self.plan.insert(0, self.info_extraction_subtask_dict)

          return None
     
     def next_subtask_to_execute(self):
          """
          Only focusing the next subtask in the plan, if not the appropriate one, then pop it and analyze the next one until the appropriate one is found.
          """
          while True:
               current_subtask = self.plan[0]
               current_subtask_str = current_subtask["subtask"] + "\n"
               necessary_actions = current_subtask.get("necessary_actions", "")
               if necessary_actions.strip() != "" and not ("not needed" in necessary_actions.lower()): # means necessary_actions is a must
                    current_subtask_str += f"**Necessary Actions**: {necessary_actions}\n"
               else:
                    current_subtask_str += f"**Target Page Description**: {current_subtask.get('target_page_description', '')}\n"

               finished_subtasks_str = ""
               for subtask in self.finished_subtasks:
                    # finished_subtasks_str += f"""{subtask.idx+1}. **Subtask**: {subtask.content}\n\t**Subtask Completeness**: {subtask.subtask_completeness}\n"""
                    finished_subtasks_str += f"""{subtask.idx+1}. **Subtask**: {subtask.content}\n"""
                    
               plan_str = ""  # construct a plan string without expectation
               plan_list = self.plan[1:]
               for idx, subtask_dict in enumerate(plan_list):
                    plan_str += str(idx+1) + ". " + subtask_dict["subtask"] + "\n"

               last_subtask_actions = next((subtask.executed_actions for subtask in reversed(self.finished_subtasks) if subtask.executed_actions != []), [])
               last_subtask_actions_str = gen_readable_actions(last_subtask_actions[1:]) # remove the first action, which is the root.src_action (noop)

               next_subtask_dict = {}

               # redundancy
               prompt = gen_next_subtask_to_execute_prompt.redundancy(domain=self.domain, maintask=self.content, expectation=self.expectation,
                                                                      current_subtask=current_subtask_str,
                                                                      finished_subtasks=finished_subtasks_str,
                                                                      last_subtask_actions=last_subtask_actions_str,)
               response = ask_LLM(prompt)
               next_subtask_dict.update(json.loads(response))
               should_be_exeucted = not str2bool(next_subtask_dict["redundant"])

               if should_be_exeucted:
                    break
               else:
                    # if not, meaning the current subtask is already completed, then pop it make it as the finished subtask
                    finished_subtask = SubTask_cls(content=current_subtask["subtask"], idx=len(self.finished_subtasks))
                    finished_subtask.executed_actions = []
                    finished_subtask.subtask_completeness = "Completed by the previous subtask."
                    if self.plan != []:
                         self.plan.pop(0)
                         self.finished_subtasks.append(finished_subtask)
                    else:
                         self.stop_flag = True
                         break
                    # enter the next cycle for analyzing the next subtask

               if self.plan == []:
                    break

          return None


     def is_webtask_stopped(self, final_actree, final_obs_des, plan_str_is_webtask_stopped, finished_subtasks_str,
                            last_subtask_is_infoextraction):
          prompt = gen_is_webtask_stopped_prompt.finished_subtasks(self.content, self.domain, self.expectation,
                                                                      finished_subtasks_str)
          response = ask_LLM(prompt)
          stop_decision_dict = json.loads(response)
          # plan
          if not last_subtask_is_infoextraction:
               prompt = gen_is_webtask_stopped_prompt.plan(self.content, self.domain, self.expectation,
                                                            plan_str_is_webtask_stopped, finished_subtasks_str)
               response = ask_LLM(prompt)
               stop_decision_dict.update(json.loads(response))
          # obserevation
          prompt = gen_is_webtask_stopped_prompt.observation(
               self.content, self.domain, self.expectation_observation, final_actree, final_obs_des)
          response = ask_LLM(prompt)
          stop_decision_dict.update(json.loads(response))
          # answer
          if self.need_answer:
               prompt = gen_is_webtask_stopped_prompt.answer(self.content, self.domain, self.expectation_answer, final_actree,
                                                             self.final_answer,)
               response = ask_LLM(prompt)
               stop_decision_dict.update(json.loads(response))

          # gather the flags, and compare them with judging results.
          is_w_s_keys = [
               str2bool(stop_decision_dict["finished_subtasks_sufficient"]),]
          if not last_subtask_is_infoextraction:
               is_w_s_keys.append(
                    str2bool(stop_decision_dict["rough_plan_is_necessary"]))
          is_w_s_keys.append(
               str2bool(stop_decision_dict["observation_meets_criteria"]))
          if self.need_answer:
               is_w_s_keys.append(str2bool(stop_decision_dict["answer_meets_criteria"]))

          if not MAJORITY_VOTE_FOR_STOP_FLAG:
               webtask_stop_reason = stop_decision_dict["reason"]
               is_stopped = str2bool(stop_decision_dict["stop_decision"])
          else:
               webtask_stop_reason = ""
               stop_1 = str2bool(stop_decision_dict.get(
                    "finished_subtasks_sufficient", False))
               if not last_subtask_is_infoextraction:
                    stop_0 = not str2bool(stop_decision_dict.get(
                         "rough_plan_is_necessary", True))
                    stop_1 = (stop_0 and stop_1)
               stop_3 = str2bool(stop_decision_dict.get(
                    "observation_meets_criteria", False))
               if self.need_answer:
                    stop_4 = str2bool(stop_decision_dict.get("answer_meets_criteria", False))
                    stop_3 = (stop_3 and stop_4)
               # if stop_1 + stop_3 >= 2 and stop_2 != False:
               if stop_1 + stop_3 >= 2:
                    is_stopped = True
               else:
                    is_stopped = False
                    webtask_stop_reason = ""
                    if stop_1 == False:
                         webtask_stop_reason += stop_decision_dict["reasoning_of_finished_subtasks"] + "\n"
                         if not last_subtask_is_infoextraction and stop_0 == False:
                              webtask_stop_reason += stop_decision_dict["reasoning_of_plan"] + "\n"
                    if stop_3 == False:
                         webtask_stop_reason += stop_decision_dict["reasoning_of_observation"] + "\n"

          return is_stopped, webtask_stop_reason

     def webtask_stop_verifier(self, actree):
          """
          More subjective, to verify the stop of the whole webtask.
          """
          detail_observation_description = self.finished_subtasks[
               -1].detail_observation_description
          finished_subtasks_str = ""
          for subtask in self.finished_subtasks:
               finished_subtasks_str += f"""{subtask.idx+1}. **Subtask**: {subtask.content}\n**Subtask Completeness**: {subtask.subtask_completeness}\n"""
          verifier_dict = {}

          if self.need_answer:
               prompt = gen_webtask_stop_verifier_prompt.answer(self.content, self.domain, self.expectation_answer, actree,
                                                                 self.ScratchPad.all_intermediate_info, self.final_answer)

               response = ask_LLM(prompt)
               verifier_dict.update(json.loads(response))

          if not MAJORITY_VOTE_FOR_STOP_FLAG:
               reason = verifier_dict["reason"]
               stop_decision = str2bool(verifier_dict["stop_decision"])

          else:
               reason = ""
               stop_decision = True

               if self.need_answer:
                    stop_3 = str2bool(verifier_dict["answer_meets_criteria"])
                    if stop_3:
                         stop_decision = True
                    else:
                         stop_decision = False
                         reason = verifier_dict["reasoning_of_answer"]

          return reason, stop_decision

     def whether_call_info_extractor(self, actree, observation_description, plan_str, finished_subtasks_str, not_stop_reason):
          """
          after webtask_stop judgment, decide whether next subtask should be info_extraction
          Input should be like gen_update_plan_prompt.
          """
          info_extraction_dict = self.info_extraction_dict

          info_needed = info_extraction_dict.get("info_needed", "")

          # subtasks_in_plan
          prompt = gen_whether_call_info_extractor_prompt.subtasks_in_plan(self.content, self.domain, self.expectation, plan_str, finished_subtasks_str,
                                                                           not_stop_reason, info_needed)
          response = ask_LLM(prompt)
          info_extraction_dict.update(json.loads(response))
          necessary_subtasks_executed = not str2bool(info_extraction_dict.get("necessary_subtask_in_plan", False))
          if not necessary_subtasks_executed:
               logging.info("The necessary subtasks are not executed (still in rough plan), so not call info_extractor and skip 'contain_info' part.")
               is_info_extraction = False
          else:
               # contain_info
               prompt = gen_whether_call_info_extractor_prompt.contain_info(self.content, self.domain, actree, observation_description, info_needed)
               response = ask_LLM(prompt)
               info_extraction_dict.update(json.loads(response))
               # contain_info_str = info_extraction_dict["reasoning_of_observation"]
               observation_contains_info = str2bool(info_extraction_dict.get("observation_contains_info", False))
               if observation_contains_info:
                    logging.info("The observation contains the info, both conditions(subtask, info) are satisfied, so call info_extractor.")
                    is_info_extraction = True
               else:
                    logging.info("The observation doesn't contain the info, so not call info_extractor.")
                    is_info_extraction = False

          # if plan is empty, task needs answer indeed, and the finished subtasks have no info_extraction subtask, then call info_extractor
          if self.plan == [] and self.need_answer and not any([subtask.interaction_type == "info_extraction" for subtask in self.finished_subtasks]):
               is_info_extraction = True
               logging.info("===== The plan is empty, and the task needs answer indeed, and the finished subtasks have no info_extraction subtask, then force to call info_extractor. However, it is not really as expected, since we hope the whether_call function can make the right decision. =====")

          subtask_dict = self.info_extraction_subtask_dict
          not_info_extraction_reason = ""
          if not observation_contains_info:
               not_info_extraction_reason += info_extraction_dict["reasoning_of_observation"]
          if not necessary_subtasks_executed:
               not_info_extraction_reason += info_extraction_dict["reasoning_of_plan"]

          return is_info_extraction, subtask_dict, not_info_extraction_reason

     def generate_final_answer(self, final_actree):
          """
          Once self.stop_flag is True, generate the final answer
          """
          if self.need_answer == False:
               self.final_answer = ""
               return None
          prompt = gen_final_answer_prompt(self.content, self.domain, final_actree, self.final_answer)

          response = ask_LLM(prompt)
          final_answer_dict = json.loads(response)

          if str2bool(final_answer_dict["answer_satisfied"]):
               pass
          else:
               self.final_answer = final_answer_dict["refined_answer"]

          return None
