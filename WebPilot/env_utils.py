# this script includes the env-related functions
import os
import re
import json
import logging
from difflib import SequenceMatcher
from urllib.parse import urlparse

from browser_env import (
    create_id_based_action,
)

from WebPilot.model import ask_LLM

# nbh_range setting
NBH_RANGE = 1

# ==================== import prompts ====================
# prompts
from WebPilot.prompter.planner import *
from WebPilot.prompter.controller import *
from WebPilot.prompter.extractor import *
from WebPilot.prompter.executor import *
from WebPilot.prompter.appraiser import *
from WebPilot.prompter.verifier import *


# ==================== action id transfer ====================
def action_id_transfer(action_dict, actree) -> dict:
     """
     for the current obs (actree), use the saved element_nbh to find the corresponding target element
     """
     action_type = action_dict["action_type"]

     # other action types don't need id
     if not action_type in ["click", "hover", "type"]:
          return action_dict

     element_nbh = action_dict["element_nbh"]

     # find the target element_id using difflib.SequenceMatcher
     max_ratio = 0

     actree_lines = actree.split("\n")

     nbh_range = NBH_RANGE
     # for each line, get the similarity score
     for line_idx, line in enumerate(actree_lines):
          nbh_lines = "" # initial every time
          # construct the nbh_lines
          if line_idx - nbh_range >= 0 and line_idx + nbh_range < len(actree_lines):
               for i in range(line_idx - nbh_range, line_idx + nbh_range + 1):
                    nbh_lines += actree_lines[i] + "\n"
          else:
               nbh_lines = line
          
          # remove the number in []
          nbh_lines = re.sub(r'\[\d+\]', '', nbh_lines)
          
          # get the similarity score
          similarity_score = SequenceMatcher(None, nbh_lines, element_nbh).ratio()
          if similarity_score > max_ratio:
               max_ratio = similarity_score
               target_line = line

     # get the element_id
     element_id = re.search(r'\[(\d+)\]', target_line).group(1)

     # update the element_info
     element_info =re.sub(r'\[\d+\]', "", target_line)
     action_dict["element_info"] = element_info

     # replace the former element_id with the new element_id
     if action_type in ["click", "hover"]:
          action_dict["action_str"] = element_id
     elif action_type == "type":
          action_dict["action_str"][0] = element_id
     
     return action_dict

# ==================== single observation description ====================
def gen_single_observation_description(task_content, actree,):
     """
     generate the description of the current observation for an actree without comparison with the former one
     """
     prompt = gen_single_observation_description_prompt(task_content, actree)
     response = ask_LLM(prompt)
     obs_des_dict = json.loads(response)

     return obs_des_dict                                                        

# ==================== observation description ====================
def gen_observation_description(actree, former_actree, WebTask, SubTask, src_action, url_changed:bool):
     """
     generate the description of the current observation for a new entered state
     works as a validation.

     Output:
          - obs_des_with_action_fulfillment: the description of the current observation with the action fulfillment
          - action_effect: the changes of the current observation
          - obs_des: the description of the current observation, without the action fulfillment
     """
     prompt = gen_observation_description_prompt.des_n_changes(domain=WebTask.domain,
                                                 former_actree=former_actree,
                                                 current_actree=actree,
                                                 is_new_page=url_changed)
     response = ask_LLM(prompt)
     obs_des_dict = json.loads(response)
     obs_des = obs_des_dict["description"]
     obs_changes = obs_des_dict["changes"]
     prompt = gen_observation_description_prompt.action_intent_fulfillment(WebTask.domain, obs_des, obs_changes,
                                                                           src_action["action_intent"],)
     response = ask_LLM(prompt)
     obs_des_dict.update(json.loads(response))

     return obs_des_dict


# ==================== back action ====================
# maybe the current & former actree are not needed
def gen_back_action(action_dict:dict, current_url:str, former_url:str) -> dict:
     # based on the former action, generate the action w.r.t currentstate, to go back to the former state
     # initial
     action_type = "noop"
     action_str = "None"
     former_action_type = action_dict["action_type"]
     former_action_str = action_dict["action_str"]
     match former_action_type:
          case "noop":
               pass
          case "click":
               # if url changed -> go_back, doesn't really solve the problem since there may also be some other action in the former page.
               if current_url != former_url:
                    action_type = "goto"
                    action_str = f"[{former_url}]"
               # if not changed, 2 situations:1). 2). the same page -> refresh page, i.e. goto [current url]
               else: 
                    action_type = "goto"
                    action_str = f"[{current_url}]"
          case "hover":
               action_type = "goto"
               action_str = current_url
          case "type":
               action_type = "type"
               type_text = re.match(r'\]\[(.*?)\]\[', former_action_str).group(1)
               action_str = former_action_str.replace(type_text, "")
          case "press":               
               pass 
          case "scroll":
               action_type = "scroll"
               action_str = "up" if former_action_str == "down" else "down"
          case "tab_focus":
               action_type = "tab_focus"
               action_str = 1 # former_page_number
          case "new_tab":
               action_type = "close_tab"
          case "close_tab":
               action_type = "press"
               action_str = "ctrl + shift + t"
          case "go_back":
               action_type = "go_forward"
          case "go_forward":
               action_type = "go_back"
          case "goto":
               action_type = "go_back"

     # construct the back_action_dict
     action_dict["action_type"] = action_type
     action_dict["action_str"] = action_str

     return action_dict

# ==================== readable_actions ====================
def gen_readable_actions(actions_list:list, with_intent=False, with_effect=True,
                         with_intent_fulfillment=False) -> str:
     """
     convert the a list of actions to readable format
     The input can also be a single action dict
     """

     if actions_list == None:
          return ""

     if not isinstance(actions_list, list):
          actions_list = [actions_list]

     readable_actions = ""
     # for action_dict in actions_list:
     for i, action_dict in enumerate(actions_list):
          if len(actions_list) > 1: # if there are more than 1 actions, then add the number
               action_description = f"Action {i+1}: \n"
          else:
               action_description = "Action: \n"
          action_description += f"""\t**Action**: [action_type]: {action_dict["action_type"]}, """
          if action_dict["action_type"] == "type": # only type action need to show the action_str
               action_description += f"""([input_text]: {action_dict["action_str"][1]}) into """
          if "element_info" in action_dict:
               cleaned_element_info = action_dict["element_info"].strip('\t')
               # action_description += f"""{cleaned_element_info} """
               action_description += f"""[element_info]: {action_dict["element_info"]}, """
          if action_dict["action_type"] == "type":
               action_description += f"""[ending_enter]: {bool(int(action_dict["action_str"][2]))} """
          if with_intent and "action_intent" in action_dict:
               action_description += f"""\n\t**action_intent**: {action_dict["action_intent"]} """
          if with_effect and "action_effect" in action_dict:
               action_description += f"""\n\t**action_effect**: {action_dict["action_effect"]} """
          if with_intent_fulfillment and "intent_fulfillment" in action_dict:
               action_description += f"""\n\t**intent_fulfillment**: {action_dict["intent_fulfillment"]} """
          readable_actions += action_description + "\n"
     return readable_actions

# ==================== find element info ====================
def find_element_info(element_id, actree):
     """
     find the corresponding element_info of the element name in the current actree
     return the line of id in the actree
     """

     # Split the actree string into lines
     lines = actree.split('\n')

     # Find the line corresponding to the element_id
     target_line = None
     for line in lines:
          if f"[{element_id}]" in line:
               target_line = line.strip() # remove the leading and trailing spaces
               break

     # delete the number and the brackets[]
     target_line =re.sub(r'\[\d+\]', "", target_line)
     
     return target_line

# ==================== find element neighborhood ====================
def find_element_nbh(element_id, actree) -> str:
     """
     find the neighborhood of the element in the actree
     the info is used for comparison between different actree, to navigate the specific element to interact with
     """
     
     # Split the actree string into lines
     lines = actree.split('\n')

     # Find the line corresponding to the element_id
     target_line = None     
     for line_idx, line in enumerate(lines):
          if f"[{element_id}]" in line:
               target_line = line.strip() # remove the leading and trailing spaces
               target_line_idx = line_idx
               break
     
     # regroup the neighborhood of the target_line, make sure the target_line in the middle of nbh
     nbh = ""
     nbh_range = NBH_RANGE
     if target_line:
          if (target_line_idx - nbh_range) >= 0 and (target_line_idx + nbh_range) < len(lines):
               for i in range(target_line_idx - nbh_range, target_line_idx + nbh_range + 1):
                    nbh += lines[i] + "\n"
          else:
               nbh = target_line          
     
     # remove all number in []
     nbh = re.sub(r'\[\d+\]', "", nbh)

     return nbh



# ==================== add element info ====================
def add_element_info(action_dict, actree):
     """
     add element info into action_dict from actree
     also, add element neighborhood infor to the action_dict for further usej, when faced with new actree
     """
     if not action_dict:
          return None

     action_type = action_dict["action_type"]
     match action_type:
          case "noop":
               action_dict["element_info"] = "None"
               return action_dict
          case "click":
               element_id = action_dict["action_str"]
          case "hover":
               element_id = action_dict["action_str"]
          case "type":
               element_id = action_dict["action_str"][0]
          case "press":
               action_dict["element_info"] = action_dict["action_str"]
               return action_dict
          case "scroll":
               action_dict["element_info"] = action_dict["action_str"]
               return action_dict
          case "tab_focus":
               action_dict["element_info"] = action_dict["action_str"]
               return action_dict
          case "new_tab":
               action_dict["element_info"] = ""
               return action_dict
          case "close_tab":
               action_dict["element_info"] = ""
               return action_dict
          case "go_back":
               action_dict["element_info"] = ""
               return action_dict
          case "go_forward":
               action_dict["element_info"] = ""
               return action_dict
          case "goto":
               action_dict["element_info"] = action_dict["action_str"]
               return action_dict
     
     # find the corresponding element_info when needed
     action_dict["element_info"] = find_element_info(element_id, actree)

     # find the neighborhood of the element
     action_dict["element_nbh"] = find_element_nbh(element_id, actree)
     
     logging.info("===== current process: add_element_info =====")
     log_info = "element_info and element_nbh added to the action_dict\n"
     log_info += f"[element_info]:\n {action_dict['element_info']}, \n[element_nbh]:\n {action_dict['element_nbh']}"
     logging.info(log_info)

     return action_dict


# ==================== logging_evaluation_info =====================
def logging_evaluation_info(final_answer:str, final_url:str, final_actree:str, all_executed_actions:list[dict]):
     """
     log the evaluation info for the webarena's evaluator.
     This information are for inspection by ourselves.
     """

     logging.info("===== current process: logging_evaluation_info =====")
     log_info = f"===== Final Actree =====\n{final_actree}\n"
     log_info += f"===== Final Answer =====\n{final_answer}\n"
     log_info += f"===== Final URL =====\n{final_url}\n"
     executed_action_str = ""
     for action in all_executed_actions:
          executed_action_str += f"[action_type]: {action['action_type']}, [action_str]: {action['action_str']}, [element_info]: {action['element_info']}.\n"
     log_info += f"===== All Executed Actions =====\n{executed_action_str}\n"

     logging.info(log_info)

     return None

def logging_finishing_subtask_info(terminal_node, final_answer:str, final_url:str, final_actree:str, all_executed_actions:list[dict]):
     """
     log the subtask relevant info
     """
     logging.info("===== current process: One Subtask finished, logging_finishing_subtask_info =====")
     log_info = "evaluation info:\n"
     log_info += f"===== Terminal Node: {terminal_node.name[:6]} =====\n"
     log_info += f"===== Answer =====\n{final_answer}\n"
     log_info += f"===== URL =====\n{final_url}\n"
     # log_info += f"===== Actree =====\n{final_actree}\n"
     executed_action_str = ""
     for action in all_executed_actions[1:]:
          executed_action_str += f"[action_type]: {action['action_type']}, [action_str]: {action['action_str']}, [element_info]: {action['element_info']}.\n"
     log_info += f"===== All Executed Actions =====\n{executed_action_str}\n"

     logging.info(log_info)

     return None
     
# ==================== verify env goto node =====================
def verify_env_goto_node(former_url, former_actree, current_url, current_actree):
     """
     verify whether the env_goto_node() is successfully executed
     Target is to verify whether the current state is the same as the former state
     """
     # 1. if the url is the same, then the action is not executed
     # if former_url == current_url:
     #      return True

     # 2. if the actree is the same, then the action is not executed. not possible actually
     # if former_actree == current_actree:
     #      return True
     
     # 3. compare actree
     # 3.1 first, remove all the element_id within [] at the beginning of each line
     former_actree = re.sub(r'\[\d+\]', "", former_actree)
     current_actree = re.sub(r'\[\d+\]', "", current_actree)

     # 3.2 compare the similarity of the actree
     similarity_score = SequenceMatcher(None, former_actree, current_actree).ratio()
     if similarity_score > 0.9: # for now
          return True
     else:
          return False


# ==================== trimm to avoid errors ====================
def trimm_actree(actree:str):
     if not isinstance(actree, str):
          return ""

     def remove_first_two_lines(actree):
          """
          remove the first two lines of the actree, the tab info and an empty line
          """
          actree = re.sub(r'^.*\n.*\n', "", actree)
          return actree
     actree = remove_first_two_lines(actree)
     
     def remove_contribution_elements(actree):
          # 1. detect the target string
          if "graphics-symbol 'No contributions<br />" not in actree:
               return actree
          else: 
               if actree.count("graphics-symbol 'No contributions<br />") > 3:
                    # 2. remove the whole line where the target string appears
                    actree = re.sub(r'.*graphics-symbol \'No contributions<br />.*\n', "", actree)
                    return actree
               else:
                    return actree # no need to trimm, but possiblly not happen
     actree = remove_contribution_elements(actree)

     def remove_commit_feed_link(actree):
          # remove the whole line where string: link 'Commits feed' appears
          actree = re.sub(r'.*link \'Commits feed\'.*\n', "", actree)
          return actree          
     actree = remove_commit_feed_link(actree)

     def remove_subscribe_link(actree):
          # remove the whole line where string: link 'Subscribe' appears
          actree = re.sub(r'.*link \'Subscribe\'.*\n', "", actree)
          return actree
     actree = remove_subscribe_link(actree)

     def remove_rss_feed_link(actree):
          # remove the whole line where string: link 'RSS feed' appears
          actree = re.sub(r'.*link \'Subscribe to RSS feed\'.*\n', "", actree)
          return actree
     actree = remove_rss_feed_link(actree)

     def remove_subscribe_calendar_link(actree):
          # remove the whole line where string: link 'Subscribe to calendar' appears
          actree = re.sub(r'.*link \'Subscribe to calendar\'.*\n', "", actree)
          return actree

     def remove_empty_image_lines(actree):
          # remove the whole line where string: image '' appears, but keep the "image 'something'" line
          actree = re.sub(r'^\t*\[.*\] image \'\'\n', "", actree, flags=re.MULTILINE)
          return actree
     actree = remove_empty_image_lines(actree)

     return actree

# ==================== filter url, to avoid using the url which is not directly accessible ====================
def filter_url(url) -> bool:
     """
     Some urls are not directly accessible, we need to identify them. output is a bool
     Now specifically for map domain.
     """
     is_directly_accessible = True

     return is_directly_accessible


# ==================== transfer the possible string of bool into bool ====================
def str2bool(input_str):
     """
     input could be bool or str of (True/False)
     sometimes the LM's output for a bool is a string, we need to transfer it to bool.
     """

     if repr(input_str).lower() == "true":
          return True
     else:
          return False

def print_n_log(log_str):
     """
     print and log the current process
     """
     print(log_str)
     logging.info(log_str)