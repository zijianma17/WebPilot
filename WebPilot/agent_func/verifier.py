from WebPilot.env_utils import *
from WebPilot.prompter.verifier import *

# ==================== verifier for different scenerios====================
class ActionVerifier:
     # ==================== action regularizer ====================
     @staticmethod
     def format_regularizer(actree, action_dict, former_prompt):
          """
          use the so-called AI-function to regularize the action, make it in the right format.
          Following this function should be add_element_info().
          """

          # if REGULARRIZE_FLAG is True, then regularize the action
          if REGULARIZE_FLAG:
               max_call_num = 3 # to save the cost
               well_formed = False
               call_num = 0
               while not well_formed and call_num < max_call_num:
                    well_formed = True
                    # initialize the warning list
                    warning_list = []

                    # check the key action_intent, if the dict has the key action_inten
                    try:
                         action_intent = action_dict["action_intent"]
                    except KeyError:
                         well_formed = False
                         warning_list.append("key action_intent not found, maybe in the wrong key, regenerate please.")

                    reasoning_process = action_dict["reasoning_process"]
                    element_choice = action_dict.get("element_choice", "")
                    element_choosing_warning = {"action_intent": action_intent, "reasoning_process": reasoning_process, "element_choice": element_choice}

                    # check the key action_type, if the dict has the key action_type
                    try:
                         action_type = action_dict["action_type"]
                    except KeyError:
                         well_formed = False
                         warning_list.append("No action_type in the output, regenerate please.")

                    # check the action_type in predefined list
                    try:
                         assert action_type in PredefinedActionTypes
                    except AssertionError:
                         well_formed = False
                         warning_list.append(f"action_type {action_type} not in the predefined list, regenerate please.") 

                    # check the key action_str, if the dict has the key action_str
                    try:
                         assert "action_str" in action_dict
                         # action_str might be a dict incluing a inner key "action_str"
                         if isinstance(action_dict["action_str"], dict):
                              try:
                                   action_str = action_dict["action_str"]["action_str"]
                                   action_dict["action_str"] = action_str
                              except:
                                   pass
                    except AssertionError:
                         well_formed = False
                         warning_list.append("No action_str in the output, regenerate please.")
                    
                    # check element_id in action_str, only for "click", "hover", "type"
                    if action_type in ["click", "hover", "type"]:
                         if action_type in ["click", "hover"]:
                              element_id = action_dict["action_str"]
                         else: # for "type"
                              # deal with the action_str, which maybe a string, convert to specific format
                              action_str_0 = action_dict["action_str"]
                              if isinstance(action_str_0, str):
                                   action_str_0 = action_str_0.strip("[]")
                                   try:
                                        action_str = ast.literal_eval(action_str_0)
                                        action_str = list(action_str)
                                   except:
                                        try:
                                             parts = action_str_0.split(", ")
                                             id = int(parts[0])
                                             type_str = parts[1]
                                             try:
                                                  enter_key = int(parts[2])
                                             except:
                                                  enter_key = 1
                                             action_str = [id, type_str, enter_key]
                                        except:
                                             action_str = action_str_0
                              else:
                                   action_str = action_str_0
                              if len(action_str) == 2:
                                   action_str.append(1) # add the enter_key, default is 1

                              action_dict["action_str"] = action_str
                              element_id = action_dict["action_str"][0]

                         try:
                              # possible types(errors): id, "id", "[id]", [id], ["id"], ["[id]"]
                              # clean the [] and "" after repr()
                              element_id = re.sub(r'[\[\]\'\"]', "", repr(element_id))
                              if element_id.isdigit():
                                   # if element_id.isdigit(), then it is a number, then it is well_formed, transfer back to action_dict
                                   if action_type in ["click", "hover"]:
                                        action_dict["action_str"] = element_id
                                   else:
                                        action_dict["action_str"][0] = element_id
                              else: 
                                   raise ValueError
                         except:
                              well_formed = False
                              warning_list.append(f"Your element_id is [{element_id}], and it should be a pure number! Maybe you can find it in the actree, regenerate please. ")

                    # check if the current actree has the target element_id
                    if action_type in ["click", "hover", "type"]:
                         element_id = action_dict["action_str"] if action_type in ["click", "hover",] else action_dict["action_str"][0]
                         if not f"[{element_id}]" in actree:
                              well_formed = False
                              warning_list.append(f"Your element_id [{element_id}] is not found in the current actree, regenerate please.")

                    if action_type == "type":
                         # convert the input_text to a string
                         input_text = action_dict["action_str"][1]
                         if isinstance(input_text, list):
                              # remove the possible [](might be a list) in the input_text
                              input_text = repr(input_text).strip("[]")
                              action_dict["action_str"][1] = input_text
                         if action_dict["action_str"][1].strip() == "":
                              well_formed = False
                              warning_list.append("The type_text is empty, regenerate please.")

                         # check the action_str[2], if it is True/False, change it to 1/0
                         enter_key = action_dict["action_str"][2]
                         if type(enter_key) != int:
                              # true or 1 -> 1, false or 0 -> 0
                              if repr(enter_key).lower() == "true" or repr(enter_key).lower() == "1":
                                   action_dict["action_str"][2] = 1
                              elif repr(enter_key).lower() == "false" or repr(enter_key).lower() == "0":
                                   action_dict["action_str"][2] = 0
                    
                    # check the alignment of element_id and element_info
                    if action_type in ["click", "hover", "type"]:
                         element_id = action_dict["action_str"] if action_type in ["click", "hover",] else action_dict["action_str"][0]
                         try:
                              element_info = action_dict["element_info"]
                              unaligned_warning = ActionVerifier.element_alignment_verifier(element_id, element_info, actree, element_choosing_warning)
                              if unaligned_warning != "":
                                   well_formed = False
                                   warning_list.append(unaligned_warning)
                         except:
                              well_formed = False
                              warning_list.append("No element_info in the output, regenerate please.")                         

                    if well_formed:
                         print("format_regularizer call numbers: ", call_num) # print the current call_num
                         return action_dict
                    
                    # use collected warning_list to refine the action_dict
                    logging.info("===== Calling action.regularizer =====")
                    logging.info(f"The action is not well-formed, warnings:\n{warning_list}")

                    # use special prompt to refine
                    action_dict_str = ""
                    for key, value in action_dict.items():
                         action_dict_str += f"[{key}]: {value}\n"
                    prompt = gen_format_regularizer_prompt(actree, action_dict_str=action_dict_str, warnings="".join(warning_list))
                    response = ask_LLM(prompt)
                    action_dict.update(json.loads(response))

                    call_num += 1

               if call_num >= max_call_num:
                    logging.info("Attention! format_regularizer call numbers exceed the max_call_num, return a noop action.")
                    action_dict = {"action_type": "noop", "action_str": "None", "action_intent": "Failed in format_regularizer. Don't consider this action."}
                    
                    return action_dict
          
          return None
     
     # ==================== element-id alignment verifier ====================
     @staticmethod
     def element_alignment_verifier(element_id, element_info, actree, element_choosing_warning):
          """
          Generated element_info may not align with element_id. Verifiy it. Call the llm to re-generate if needed.

          Args:
               element_id (str): The element ID to verify.
               element_info (str): The generated element info.
               actree (str): The actree string.
               element_choosing_warning (dict): reasoning_process, action_intent, element_choice

          Returns:
               str: Warning about the misalignment.          
          """

          # use the info to get the line in the actree
          def find_info_line(info, actree):
               # get the highest similarity line
               max_score = 0
               max_line = ""
               for line in actree.split("\n"):
                    # delete the [id] in the line
                    line_without_id =re.sub(r'\[\d+\]', "", line)
                    score = SequenceMatcher(None, line_without_id, info).ratio()
                    if score > max_score:
                         max_score = score
                         max_line = line
               print(f"max_score: {max_score}, max_line: {max_line}")
               return max_line
          info_line = find_info_line(element_info, actree)

          # use the id to get the line in the actree
          def find_id_line(id, actree):
               for line in actree.split("\n"):
                    if f"[{id}]" in line:
                         return line
               return ""
          id_line = find_id_line(element_id, actree)

          aligned = True if SequenceMatcher(None, id_line, info_line).ratio() > 0.9 else False

          # # clean the line to only include the text without any punctuation
          cleaned_line = id_line.strip("\n\t")
          cleaned_line = re.sub(r'[^\w\s]', '', cleaned_line).lower()
          cleaned_element_info = element_info.strip("\n\t")
          cleaned_element_info = re.sub(r'[^\w\s]', '', cleaned_element_info).lower()

          # if element_info doen't align with any lines in the actree, then it is non-exist info, we take it as aligned -> use the id directly
          def info_exist(element_info, actree):
               for line in actree.split("\n"):
                    cleaned_line = re.sub(r'[^\w\s]', '', line).lower()
                    if element_info in cleaned_line:
                         return True
               return False
          def info_exist_info_line(element_info, info_line):
               cleaned_info_line = re.sub(r'[^\w\s]', '', info_line).lower()
               if element_info in cleaned_info_line:
                    return True
               return False
          aligned = aligned or not (info_exist(cleaned_element_info, actree) and info_exist_info_line(cleaned_element_info, info_line))

          if cleaned_element_info in cleaned_line:
               aligned = True
                         
          # check whether the element_info is in the cleaned line 
          if not aligned:
               logging.info(f"Attention! The element_info {element_info} is not aligned with the element_id {element_id}. Re-generate the action.")
               warning = gen_element_info_alignment_warning_prompt(element_choosing_warning["reasoning_process"],
                                                                 element_choosing_warning["action_intent"],
                                                                 element_choosing_warning["element_choice"],
                                                                 id_line,
                                                                 info_line,
                                                                 )
               return warning
          else:
               return ""
          
     # ==================== action verifier ====================
     @staticmethod
     def interact_verifier(action_dict, former_actree, former_url, actree, url):
          """
          Given the former_actree, actree, former_url, url, compare and verify the current action, whether the action is indeed executed.
          """
          execution_reflection = ""

          # 1. if url changed, the action is very likely executed
          if former_url != url:
               return True, execution_reflection
          
          # 2. if the string similarity between former_actree and actree is low, then the action is probably executed
          similarity_score = SequenceMatcher(None, former_actree, actree).ratio()
          # the threshold is hard to define, but 0.5 means at least half of the actree is changed
          if similarity_score < 0.5:
               return True, execution_reflection
          
          # 3. compare the actree
          if former_actree == actree:
               logging.info("Attention! The action doesn't change the actree, env remains the same, the action is not executed.")
               simple_action_dict = {"action_type": action_dict["action_type"],
                                   "action_str": action_dict["action_str"],
                                   "action_intent": action_dict["action_intent"]}
               prompt = gen_execution_reflection_prompt(action=repr(simple_action_dict),
                                                       former_actree=former_actree,
                                                       actree=actree,
                                                       )
               response = ask_LLM(prompt)
               execution_reflection = json.loads(response)["execution_reflection"]          

               return False, execution_reflection


          return True, execution_reflection
     
     # ==================== action deduplicator ====================
     @staticmethod
     def action_deduplicator(actree, action_dict, former_prompt, sibling_actions:list,):
          """
          in gen_next_action_with_reflection, the generated action should not be same as its siblings.
          This function is used to ensure it and generate a new action.
          """
          action_duplicated = True
          max_call_num = 3
          call_num = 0
          while action_duplicated and call_num < max_call_num:
               call_num += 1
               action_duplicated = False
               # 1. check the generated action with its siblings' actions
               for sibling_action in sibling_actions:
                    if action_dict["action_type"] != sibling_action["action_type"] or action_dict["action_type"] == "scroll":
                         continue
                    if "Failed in format_regularizer" in action_dict["action_intent"]:
                         continue
                    # only "click", "hover", "type" need to compare the element_nbh
                    if action_dict["action_type"] in ["click", "hover", "type"]:
                         # compare the similarity of the element_nbh
                         new_nbh = action_dict["element_nbh"]
                         sibling_nbh = sibling_action["element_nbh"]
                         # clear the \t
                         new_nbh = re.sub(r'\t', "", new_nbh)
                         sibling_nbh = re.sub(r'\t', "", sibling_nbh)
                         similarity_score = SequenceMatcher(None, new_nbh, sibling_nbh).ratio()
                         if similarity_score > 0.9: # 0.9 for now
                              action_duplicated = True
                              simple_same_action = {"action_type": action_dict["action_type"],
                                                  "action_str": action_dict["action_str"],
                                                  "element_info": action_dict["element_info"],
                                                  "action_intent": action_dict["action_intent"]}
              
                    # other actions, compare the action_str
                    else:
                         if action_dict["action_str"] == sibling_action["action_str"]:
                              action_duplicated = True
                              simple_same_action = {"action_type": action_dict["action_type"],
                                                       "action_str": action_dict["action_str"],
                                                       "action_intent": action_dict["action_intent"]}

                    if action_duplicated:
                         logging.info("===== current process: ActionVerifier.action_deduplicator is called =====")
                         avoid_action_prompt = f"""
### Same Action
Attention: The following action is the same as former trials, don't generate the same action again:
{repr(simple_same_action)}

"""                      
                         logging.info("Attention! The action is the same as its siblings, re-generate the action.")
                         break

               # 2. if the action is the same as its siblings, then use the former prompt to re-generate the action
               if action_duplicated:
                    prompt = former_prompt + avoid_action_prompt
                    response = ask_LLM(prompt)

                    action_generation_dict = json.loads(response)
                    action_dict = ActionVerifier.format_regularizer(actree, action_generation_dict, prompt)
                    action_dict = add_element_info(action_dict, actree)

          # if max_call_num is reached, then return the action
          if call_num >= max_call_num:
               logging.info(f"===== Attention! action_deduplicator call numbers exceed the max_call_num({max_call_num}), this node will be deleted. =====")
               return f"DELETE_NODE: {action_dict['action_type']} {action_dict['action_str']}"
          
          action_dict["gen_action_prompt"] = former_prompt
          return action_dict


# ==================== execution reflection ====================
def gen_general_execution_reflection(new_action, failed_action, failed_execution_reflection):
     """
     if an action not successfully executed at the first time, there is a failed_execution_reflection.
     After verification, we generate a successful_execution_reflection, ant combine them to a general_execution_reflection
     """

     # get simplified action dict
     simple_new_action = {"action_type": new_action["action_type"],
                          "action_str": new_action["action_str"],
                          "action_intent": new_action["action_intent"]}
     simple_failed_action = {"action_type": failed_action["action_type"],
                             "action_str": failed_action["action_str"],
                             "action_intent": failed_action["action_intent"]}
     
     prompt = gen_general_execution_reflection_prompt(new_action=repr(simple_new_action),
                                                     failed_action=repr(simple_failed_action),
                                                     failed_execution_reflection=failed_execution_reflection,
                                                     )
     response = ask_LLM(prompt)
     general_execution_reflection = json.loads(response)["general_execution_reflection"]

     return general_execution_reflection


