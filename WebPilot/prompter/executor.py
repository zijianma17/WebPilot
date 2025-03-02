import logging
from .prompts import *

# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with automating interactions on a webpage. """
AGENT_ROLE = "executor"

maintask_prompt = """### Main Task
Here is the main task you need to complete:
{task_content}

"""

# ========== redefine prior_knowledge_prompt =========
def gen_prior_knowledge_prompt(domain):
    return gen_prior_knowledge_prompt_origin(domain, AGENT_ROLE)

# ========== basic prompt with prior knowledge =========
def gen_basic_prompt(task_content, domain, agent_role=AGENT_ROLE, function_description=""):
    return gen_basic_prompt_origin(task_content, domain, agent_role, function_description)

# ========== action spaces ==========
PredefinedActionTypes = ["noop", "click", "hover", "type", "press", "scroll", "tab_focus", "new_tab", "close_tab", "go_back", "go_forward", "goto"]
def gen_action_space_prompt(only_info_extraction_flag:bool = False):
    action_space_prompt = """### Action Space
A well-formed action comprises an "action_type" and an "action_str". Below is a list of the available action types, their expected action strings (`action_str`), and a brief description of each action:
**click**: {"action_type": "click", "action_str": element_id(int), "element_info": element_name(str)}: click on the specific element with the id, its name is the 'element_name';
**hover**: {"action_type": "hover", "action_str": element_id(int), "element_info": element_name(str)}: hover on the specific element with the id, its name is the 'element_name';
**type**: {"action_type": "type", "action_str": [element_id(int), "text"(str), 0|1(int)], "element_info": textbox_name(str)}: input the text to the specific element with the id(do not give empty text), 0|1 means whether click ENTER key after input True|False;
**scroll**: {"action_type": "scroll", "action_str": "down|up"(str), "element_info": None}: scroll the page up or down for more information, no need to specify the element_id and element_info;

"element_id(int)" is an integer found in brackets '[]' in the observation, uniquely identifying webpage elements for interaction.
For effective execution, please ensure actions are properly formatted according to these guidelines.

"""
    if only_info_extraction_flag:
        action_space_prompt += """The current "Main Task" only needs to extract information without interacting with the webpage. So consider only using "scroll" action to control the page to present the information you need.\n\n"""
    return action_space_prompt  

# currently delete following actions, since fow now they are not so useful
"""
examples:
- e.g. {"action_type": "click", "action_str": 123, "element_info": "link 'Home'"}: click on the link 'Home', its element_id is 123;
- e.g. {"action_type": "hover", "action_str": 456, "element_info": "menuitem 'Clothes'"}: hover on the menuitem 'Clothes', its element_id is 456;
- e.g. {"action_type": "type", "action_str": [456, "Hello World", 1], "element_info": "textbox 'Search'"}: input 'Hello World' to the textbox 'Search', its element_id is 456, and press ENTER key;
- e.g. {"action_type": "scroll", "action_str": "down",}: scroll the page down.

**noop**:
- {"action_type": "noop", "action_str": ""}
- take no action, means the current state is good enough;
{{"action_type": hover, "action_str": element_id}}, hover on the specific element with the id;
{{"action_type": press, "action_str": key_combination}}, press a key combination (e.g. ctrl + v);
**tab_focus**:
- {"action_type": "tab_focus", "action_str": "page_number"}
- bring the open tab with the page number to the front;
**new_tab**:
- {"action_type": "new_tab", "action_str": ""}
- open a new tab;
**close_tab**:
- {"action_type": "close_tab", "action_str": ""}
- close the current tab and focus on the last open page;
**goto**:
- {"action_type": "goto", "action_str": "URL"}
- go to a URL on the current page;
**go_back**:
- {"action_type": "go_back", "action_str": ""}
- go back to the previous page(last url);
**go_forward**:
- {"action_type": "go_forward", "action_str": ""}
- undo go back operation;
"""

# ========== current observation ==========
# here in executor, this is redefined, with the explanation of Indentation, In-page Tabs, Dropdown Menus, Top/Side Bars
observation_prompt = """### Observation
The following is the current state of the webpage:
{actree}

In this structure, each line details a web element with its attributes, formatted for clarity:
    - `[element_id]`: A unique identifier for the element, represented as an integer. This ID is used to specify actions targeting the element.
    - `element_type`: Describes the element's role (e.g., 'button', 'link', 'StaticText', 'textbox'). Buttons are typically clickable; links navigate to other pages; StaticText elements are static and cannot be interacted with; textboxes allow text input.
    - `element_text`: Displays the visible text associated with the element. This text is usually static but can be dynamic for textboxes, reflecting their current value.

[Indentation] The structure also incorporates indentation to show parent-child relationships among elements. An indented element is considered a child of the directly preceding less-indented element, illustrating the hierarchy of the webpage's content.
[In-page Tabs] The page may contain multiple categorized tabs, but only the tab with "focused: True" is in the open state, indicating that you are currently under this tab.
[Dropdown Menus] Elements with "hasPopup: menu|listbox" are dropdown menus. The "expanded: True|False" attribute indicates whether the dropdown menu is open or closed, followed by its child elements. An open dropdown menu means you have to interact with it to select an option.
[Top/Side Bars] The top bar elements appear at the top of the actree, followed by sidebar elements. The main body of the actree typically appears after the 'main' element. If sidebar elements are clearly visible, the current interface may be related to a specific project.

"""
observation_without_explanation_prompt = """### Observation
The following is the current state of the webpage:
{actree}

"""

avoid_repeating_action_prompt = """Important: Carefully review these actions to ensure diversity in your approach. You MUST NOT repeat the same action, because it's unnecessary and may indicate a loop in the process. If you still think it is necessary to repeat them, provide your reasoning.
An exception is when the executed action is "scroll", which may be useful to be repeated to explore more information on the page.

"""

# ==================== next_action ====================
def gen_next_action_prompt(WebTask, SubTask,
                            finished_subtasks,
                            task_content, domain, actree,
                            executed_actions,  
                            parent_node_sim_reflection, parent_node_reflection_for_child,
                            not_stop_reason:str = "", # for the nodes who decide not to stop in gen_action_with_stop_asking
                            last_subtask_reflection:str = "", # will be none empty when the last subtask is not completed
                            ):
    """
    Generate the prompt for next action generation. Only one action, mostly used in simulation.
    """
    logging.info("===== Currentprompt: gen_next_action_prompt =====")
    
    #special prompts
    function_description = """To achieve the task goal, you need to generate the next action to interact with the webpage according to the given information. \n"""
    finished_subtasks_prompt = f"""### Finished tasks
Besides the current "Main Task", you have completed the following tasks:
{finished_subtasks}Based on the completed subtasks, don't duplicate the work you've done.

"""
    attention_prompt = """Attention: Use element_id, the output should be an integer.
"""
    parent_sim_reflection_prompt = f"""### Simulation Reflection
Here is the reflection of the simulation, where you have tried the same situation before:
{parent_node_sim_reflection}
This reflection may contain some judgments or insights, which can be helpful for your decision-making.

"""
    parent_node_reflection_for_child_prompt = f"""### Parent Node Reflection
Here is the reflection generated from your last state, which may contain some insights for further actions:
{parent_node_reflection_for_child}

"""
    not_stop_reason_prompt = f"""### Reason for not stopping
In the current situation, you decide not to stop the exploration process. Here is the reason you gave:
{not_stop_reason}
Consider the reason, which could include some reflection about what you should do next.

"""
    last_subtask_reflection_prompt = f"""### Last Time Reflection
Here is the analysis of your previous trial on the "Main Task", think about where you are, try to avoid the same mistake:
{last_subtask_reflection}

"""

    # conditionally add the keywords to the output_prompt
    output_prompt = """### Reasoning Process\nConsider the "Main Task", """
    if executed_actions.strip() != "":
        output_prompt += """what you've done in "Executed Actions", """
    if not_stop_reason.strip() != "":
        output_prompt += """the "Reason for not stopping", """
    if parent_node_sim_reflection.strip() != "":
        output_prompt += """the "Simulation Reflection", """
    if parent_node_reflection_for_child.strip() != "":
        output_prompt += """the "Parent Node Reflection", """
    if last_subtask_reflection.strip() != "":
        output_prompt += """the "Last Time Reflection". """
    if finished_subtasks.strip() != "":
        output_prompt += """The "Finished tasks" are what you have completed, don't duplicate the work you've done. """    
    output_prompt += """What is your next step?\n"""
    output_prompt += """Interact with the element from "Observation". Output your detailed reasoning process for your next action after the key "reasoning_process".
Identify the element you want to interact with. If there are multiple elements with the same name, list them all, analyze their potential outcomes, and choose the most appropriate one. Present your analysis and choice under the key “element_choice”.
Next, summarize your “action_intent” by explaining what you expect to achieve with the action in a concise sentence.
Based on your analysis, specify the “action_type” and “action_str” that align with your intended action. Ensure that “action_type” corresponds to the type of interaction, “element_info” accurately reflects the element you chose to interact with, and “action_str” is correctly formatted according to the action space description.

"""
    format_instructions_prompt = """Format your response in Json, including the following:
{   
    "reasoning_process": string that describes your reasoning process for determining the next action, ensuring the chosen element aligns with the 'Main Task';
    "action_intent": string that outlines your expectations for the outcome of the action, ensuring it is consistent with the task's objectives and the selected element;
    "element_choice": string that compares the elements you want to interact with (if they have the same name but different element IDs), predicts their possible effects, justifies your choice, and ensures it aligns with the 'reasoning_process' and 'Main Task';
    "action_type": string that specifies the type of action, chosen from the predefined action types;
    "element_info": string that includes the information after the element_id in the observation (i.e., the text associated with the element), ensuring it matches the element identified in the 'reasoning_process' and 'element_choice';
    "action_str": a STRING of the action-specific annotations. Integer element_id for 'click', 'hover', list of [id, text, 0|1] for 'type', down|up for 'scroll'.
}
"""

    prompt = basic_prompt_without_info
    prompt += function_description
    prompt += observation_prompt.format(actree=actree)
    prompt += gen_action_space_prompt(SubTask.only_info_extraction_flag)
    if finished_subtasks.strip() != "":
        prompt += finished_subtasks_prompt
    prompt += maintask_prompt.format(task_content=task_content)
    if executed_actions.strip() != "":
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
        prompt += avoid_repeating_action_prompt
    if parent_node_reflection_for_child.strip() != "":
        prompt += parent_node_reflection_for_child_prompt
    if parent_node_sim_reflection.strip() != "":
        prompt += parent_sim_reflection_prompt
    if not_stop_reason.strip() != "":
        prompt += not_stop_reason_prompt
    if last_subtask_reflection.strip() != "":
        prompt += last_subtask_reflection_prompt
    prompt += output_prompt
    prompt += attention_prompt
    prompt += gen_prior_knowledge_prompt(domain)
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt


# ==================== next_action with sibling reflections ====================
def gen_next_action_with_reflection_prompt(WebTask, SubTask,
                                           finished_subtasks,
                                           task_content, domain, actree,
                                           executed_actions, 
                                           sibling_actions_n_reflections:list[str],
                                           parent_node_sim_reflection, parent_node_reflection_for_child,
                                           not_stop_reason:str = "",
                                           last_subtask_reflection:str = "",
                                           ):
    """
    Generate the prompt for next action generation. Only one action, mostly used in simulation.
    """
    logging.info("===== Currentprompt: gen_next_action_with_reflection_prompt =====")
    
    # speical prompts
    function_description = """To achieve the task goal, you need to generate the next action to interact with the webpage according to the given information. \n"""
    finished_subtasks_prompt = f"""### Finished tasks
Besides the current "Main Task", you have completed the following tasks:
{finished_subtasks}Based on the completed subtasks, don't duplicate the work you've done.

"""
    sibling_actions_n_reflections_prompt = """### Sibling Actions and Reflections
You are doing an MCTS search, and you have been the current situation previously, where you have tried different actions and made reflections on them. They can be seen as your siblings. Here are the actions and corresponding reflections:
"""
    for i in range(len(sibling_actions_n_reflections)):
        sibling_actions_n_reflections_prompt += f"""**Sibling {i+1}**:\n {sibling_actions_n_reflections[i]}\n"""
    
    # if all terms in list is not "", then sibling_reflection is valid
    if all([sibling_actions_n_reflections[i].strip() != "" for i in range(len(sibling_actions_n_reflections))]):
        sibling_valid_flag = True

    sibling_actions_n_reflections_prompt += """
Remember, these actions are NOT executed!
You MUST NOT repeat the "Sibling Actions" and try something new to explore the environment. Take look at the Reflections, they may contain some judgments or insights that describe the effect of their corresponding actions.

"""
    attention_prompt = """Attention: Use element_id, the output should be an integer. Do not repeat the actions in "Sibling Actions"."""
    parent_sim_reflection_prompt = f"""### Simulation Reflection
Here is the reflection of the simulation, where you have tried the same situation before:
{parent_node_sim_reflection}This reflection may contain some judgments or insights, which can be helpful for your decision-making.

"""
    parent_node_reflection_for_child_prompt = f"""### Parent Node Reflection
Here is the reflection generated from your last state, i.e. on the last action in the "Executed Actions", which may contain some insights for further actions:
{parent_node_reflection_for_child} 

"""
    not_stop_reason_prompt = f"""### Reason for not stopping
In the current situation, you decide not to stop the exploration process. Here is the reason you gave:
{not_stop_reason}Consider the reason, which could include some reflection about what you should do next.

"""
    last_subtask_reflection_prompt = f"""### Last Time Reflection
Here is the analysis of your previous trial on the "Main Task", think about where you are, try to avoid the same mistake:
{last_subtask_reflection}

"""
    # conditionally add the keywords to the output_prompt
    output_prompt = """### Reasoning Process\nConsider the "Main Task", """
    if executed_actions.strip() != "":
        output_prompt += """what you've done in "Executed Actions", """
    if not_stop_reason.strip() != "":
        output_prompt += """the "Reason for not stopping", """
    if parent_node_sim_reflection.strip() != "":
        output_prompt += """the "Simulation Reflection", """
    if parent_node_reflection_for_child.strip() != "":
        output_prompt += """the "Parent Node Reflection", """
    if sibling_valid_flag:
        output_prompt += """the "Sibling Actions and Reflections", """
    if last_subtask_reflection.strip() != "":
        output_prompt += """the "Last Time Reflection", """
    if finished_subtasks.strip() != "":
        output_prompt += """The "Finished tasks" are what you have completed, don't duplicate the work you've done. """
    output_prompt += """What is your next step?\n"""
    output_prompt += """Interact with the element from "Observation". Output your detailed reasoning process for your next action after the key "reasoning_process".
Identify the element you want to interact with. If there are multiple elements with the same name, list them all, analyze their potential outcomes, and choose the most appropriate one. Present your analysis and choice under the key “element_choice”.
Next, summarize your “action_intent” by explaining what you expect to achieve with the action in a concise sentence.
Based on your analysis, specify the “action_type” and “action_str” that align with your intended action. Ensure that “action_type” corresponds to the type of interaction, “element_info” accurately reflects the element you chose to interact with, and “action_str” is correctly formatted according to the action space description.

"""
    format_instructions_prompt = """Format your response in Json, including the following:
{   
    "reasoning_process": string that describes your reasoning process for determining the next action, ensuring the chosen element aligns with the 'Main Task';
    "action_intent": string that outlines your expectations for the outcome of the action, ensuring it is consistent with the task's objectives and the selected element;
    "element_choice": string that compares the elements you want to interact with (if they have the same name but different element IDs), predicts their possible effects, justifies your choice, and ensures it aligns with the 'reasoning_process' and 'Main Task';
    "action_type": string that specifies the type of action, chosen from the predefined action types;
    "element_info": string that includes the information after the element_id in the observation (i.e., the text associated with the element), ensuring it matches the element identified in the 'reasoning_process' and 'element_choice';
    "action_str": a STRING of the action-specific annotations. Integer element_id for 'click', 'hover', list of [id, text, 0|1] for 'type', down|up for 'scroll'.
}
"""
    # prompt = gen_basic_prompt("", domain)
    prompt = basic_prompt_without_info
    prompt += function_description
    prompt += observation_prompt.format(actree=actree)
    prompt += gen_action_space_prompt(SubTask.only_info_extraction_flag)
    if finished_subtasks.strip() != "":
        prompt += finished_subtasks_prompt
    prompt += maintask_prompt.format(task_content=task_content)
    if executed_actions.strip() != "":
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
        prompt += avoid_repeating_action_prompt
    if parent_node_reflection_for_child.strip() != "":
        prompt += parent_node_reflection_for_child_prompt
    if sibling_valid_flag:
        prompt += sibling_actions_n_reflections_prompt
    if parent_node_sim_reflection.strip() != "":
        prompt += parent_sim_reflection_prompt
    if not_stop_reason.strip() != "":
        prompt += not_stop_reason_prompt
    if last_subtask_reflection.strip() != "":
        prompt += last_subtask_reflection_prompt
    prompt += output_prompt
    prompt += attention_prompt
    prompt += gen_prior_knowledge_prompt(domain)
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt


# ==================== observation description ====================
class gen_observation_description_prompt:
    """
    will be used when entering a new node, to describe the observation and src_action.
    Output including to keys: description_and_changes, action_intent_fullfillment, separatly answering 2 questions
    """
    @staticmethod
    def des_n_changes(domain, former_actree, current_actree, is_new_page:bool):
        logging.info("===== Currentprompt: gen_observation_description_prompt.des_n_changes =====")
    
        # special prompts
        former_observation_prompt = f"""### Former Observation
The observation of the former state is: 
{former_actree}

"""
        observation_description_prompt = ""
        observation_description_prompt += """### Reasoning Process
Now closely examine the current view. Consider the following aspects:
    - **Description**: Focus on the current "Observation". Identify which elements are from the top bar and which are from the sidebar (if available). Describe the main body of the webpage in detail. What information is most critical? Determine if the current page is related to a specific project or category by observing the top/side bars."""
        # describe the changes according to whether it's a new page
        if is_new_page:
            observation_description_prompt += """
    - **Changes**: The current "Observation" is a new page. Compare it with the "Former Observation" to identify the differences. Describe the new page in detail. Although the changes might be minor, I am confident that the current "Observation" page is new. Please compare it meticulously with the "Former Observation" page to identify any differences, no matter how minor. It is crucial to observe and describe all changes in detail. Output the changes in the format: "The executed action led to a new page about xxxxx".

"""
            format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "description": string describing the current observation;
    "changes": a string detailing the changes describe the new page in the format: "The executed action led to a new page about xxxxx".
}
"""
        else:
            observation_description_prompt += """
    - **Changes**: The current "Observation" remains in the same page of the "former Observation". It means some elements have changed.
        Compare the "Former Observation" with the current "Observation". Describe the changes in detail. Consider the following types of changes:
        1. Changes in element states.
        2. Textboxes that have been filled.
        3. Dropdown menus that have been expanded or closed.
        4. A dialog has appeared or disappeared.
        Output the changes in the format: "The page remains the same, but...
        1. xxx element's state has changed from xxx to xxx.
        2. xxx element has appeared(disappeared).
        3. a dropdown menu has been expanded(closed) and the new options are xxx".   
        4. xxx dialog has been opened(closed).
        
"""        
            format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "description": string describing the current observation;
    "changes": a string detailing the changes: listing the elements that have changed; if a dropdown menu has been expanded, describe the new options;
}
"""
        prompt = ""
        prompt += gen_basic_prompt("", domain)
        if former_actree.strip() != "":
            prompt += former_observation_prompt
        prompt += observation_prompt.format(actree=current_actree)
        prompt += observation_description_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt

    @staticmethod
    def action_intent_fulfillment(domain, observation_description, obseravtion_changes, src_action_intent):
        logging.info("===== Currentprompt: gen_observation_description_prompt.action_intent_fulfillment, comparison =====")
        
        # special prompts
        function_description = """You need to evaluate the current observation to determine if the intent of the former action has been fulfilled. \n"""
        action_intent_prompt = f"""### Action Intent
The intent of the former action is:
{src_action_intent}
Attention: the intent is only your expectation of the action, not the real result of the action.

"""
        obs_des_prompt = f"""### Observation Description
Here is a description generated as a summary of the current state of the webpage:
{observation_description}

"""
        changes_prompt = f"""### Changes
After executing the former action, the following changes were observed in the environment:
{obseravtion_changes}

"""
        observation_description_prompt = """### Reasoning Process
Now you have just performed an action, examine if “Action Intent” is fulfilled by the “Changes”.

"""
        format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "action_intent_fulfillment": string that concludes whether the intent of the former action is fulfilled or not, within 20 words;
}
"""

        prompt = ""
        prompt += gen_basic_prompt("", domain, function_description=function_description)
        prompt += action_intent_prompt
        prompt += obs_des_prompt
        prompt += changes_prompt
        prompt += observation_description_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt

        return prompt



# ==================== node_reflection ====================
class gen_node_reflection_prompt:
    """
    split into 2 parts: for_child, for_sib
    """
    expectation_prompt = """### Expectation
The expectation is the state that should be achieved after correctly completing the "Main Task". It will be used to determine whether the "Main Task" has been successfully completed.
{expectation}
"""

    @staticmethod
    def for_child(task_content, domain, expectation, actree, executed_actions, obs_des_changes):
        """
        executed_actions: without intent, also without effect
        """
        logging.info("===== Currentprompt: gen_node_reflection_prompt.for_child =====")
        
        # special prompts
        function_description = """You need to reflect on the observation and the executed actions to generate insights for the next step. \n"""
        obs_des_changes_prompt = f"""### Observation Description
Following is the description of the current "Observation" and the "Changes" compared to the former state:
{obs_des_changes}
"""
        if "necessary actions" in expectation.lower():
            node_reflection_instruction_prompt = """### Reasoning Process
Reflect on the "Executed Actions" you’ve taken so far.
    - Do all the "Necessary Actions" in the "Expectation" have been executed, as shown in the "Executed Actions"? If yes, output 'no further action needed' after the key "node_reflection_for_child".
    - Otherwise, focus on the "Main Task" and "Necessary Actions". Consider what you have done so far as the "Executed Actions". What could be the next step? Your advice MUST be really able to interact with the web page, e.g. to click some relevant elements, or type some text into some boxes. Don't output something like locate, verify, confirm who look like logic thinking steps. 
    - If the "Changes" in the "Observation Description" shows some newly appeared elements, e.g. some items, links, or buttons, evaluate if these elements are necessary to interact with to fulfill all the requirements.
    Output insights or reasoning after the key "node_reflection_for_child".
This self-reflection is crucial for planning the next step.

"""
            format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "node_reflection_for_child": string; If all the "Necessary Actions" have been executed, output 'no further action needed'; if not, output insights on what could be done next, should the newly appeared elements from the "Changes" be interacted with, include references to how all "Necessary Actions" were addressed if applicable, within 50 words. Only the real interactable actions should be considered.
}
"""
        else:
            node_reflection_instruction_prompt = """### Reasoning Process
Reflect on the current "Observation", the "Observation Description", and the "Executed Actions" you’ve taken so far.
    - Does the current "Observation" meet the "Target Page Description"? If yes, output 'no further action needed' after the key "node_reflection_for_child".
    - Otherwise, to achieve the "Expectatoin, what could be the next step? Your advice MUST be really able to interact with the web page, e.g. to click some relevant elements, or type some text into some boxes. Don't output something like locate, verify, confirm who look like logic thinking steps. 
    - If the "Changes" in the "Observation" shows some newly appeared elements, e.g. some items, links, or buttons, evaluate if these elements are necessary to interact with to fulfill all the requirements.
    Output insights or reasoning after the key "node_reflection_for_child".
This self-reflection is crucial for planning the next step.

"""
            format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "node_reflection_for_child": string; If the current state meets the "Expectation", output "no further action needed"; if not, output advices on what action could be done next, should the newly appeared elements be interacted with, within 50 words. Only the real interactable actions should be considered.
}
"""
        prompt = gen_basic_prompt("", domain, function_description=function_description)
        if not "necessary actions" in expectation.lower(): # only add the observation description when not necessary actions
            prompt += observation_prompt.format(actree=actree)
        prompt += obs_des_changes_prompt
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += gen_node_reflection_prompt.expectation_prompt.format(expectation=expectation)
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
        prompt += node_reflection_instruction_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt

    @staticmethod
    def for_sibling(task_content, domain, expectation, actree, executed_actions, src_action_with_intent, obs_des_fulfillment):
        """
        executed_actions: without intent, and without the final executed action
        """
        logging.info("===== Currentprompt: gen_node_reflection_prompt.for_sib =====")
        
        # special prompts
        function_description = """You need to reflect on what you have done to generate insights for yourself when you face the same situation and make better decisions. \n"""
        if executed_actions.strip() == "":
            executed_actions = "No former actions executed.\n"
            
        src_action_prompt = f"""### Final Action
The action you've just executed is:
{src_action_with_intent}
"""
        obs_des_fulfillment_prompt = f"""### Observation Description
Here is a description generated as a summary of the current state of the webpage and how the last action has fulfilled the intent:
{obs_des_fulfillment}
"""
        node_reflection_instruction_prompt = """### Reasoning Process
**Reflection for Sibling Nodes**: generating reflection for yourself when you face the same situation, to generate better action.
Focus on the "Observation Description", the "Executed Action", and the "Final Action" you’ve taken.
    - Does the "Observation Description" align with the "Expectation"? If not,
        - If the "action_intent" is incorrect, describe and output the correct "action_intent" under the key "node_reflection_for_sib".
        - If the "action_intent" is correct, analyze the "Action Intent Fulfillment":
            - If the "action_intent" is not fulfilled, identify which part of the "Action" in "Final Action" was incorrect (e.g., action_type, element_info) and describe what action should be performed. Output this reflection and concrete improvement under the key "node_reflection_for_sib".
            - If the "action_intent" is fulfilled, output 'no better action needed' under the key "node_reflection_for_sib"."""
        if "necessary actions" in expectation.lower():
            node_reflection_instruction_prompt += """
        - Is the "Final Action" one of the "Necessary Actions" in the "Expectation"? Was it supposed to be executed when in the same situation?"""
        node_reflection_instruction_prompt += """
This self-reflection is crucial for better decision-making when facing the same situation.

"""
        format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "expectation": boolean whether the "Observation Description" align with the "Expectation";
    "action_intent_appropriate": boolean whether the "action_intent" is appropriate for achieving the "Main Task" and its "Expectation", ALWAYS considering the "Prior Knowledge";
    "node_reflection_for_sib": a string that records your reflections on the current state and possible improvements when you have another chance. Provide concrete suggestions for improvement, including what you have done wrong and what you should have done instead. The reflection should reassess the former observation and determine the correct action.
}
"""
        prompt = gen_basic_prompt("", domain, function_description=function_description)
        prompt += observation_prompt.format(actree=actree)
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += gen_node_reflection_prompt.expectation_prompt.format(expectation=expectation)
        prompt += obs_des_fulfillment_prompt
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
        prompt += src_action_prompt
        prompt += node_reflection_instruction_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt

# ==================== sim_reflection ====================
def gen_sim_reflection_prompt(task_content, expectation, domain, actree, src_action:str, ScratchPad_Info, obs_des_changes_fulfillment, score_reason):
    """
    it's actually for simlation node, and will be used for real child nodes.
    """
    logging.info("===== Currentprompt: gen_sim_reflection_prompt =====")
    
    # special prompts
    function_description = """You need to reflect on the executed action to assess the alignment between the action intent and the actual action effect. \n"""
    expectation_prompt = f"""### Expectation
The expectation is the state that should be achieved after correctly completing the "Main Task". It will be used to determine whether the "Main Task" has been successfully completed.
{expectation}
"""
    src_action_prompt = f"""### Executed Action
The action you've just executed is:
{src_action}

"""
    observation_description_prompt = f"""### Observation Description
Here is a description generated as a summary of the current state of the webpage and how the last action has fulfilled the intent:
{obs_des_changes_fulfillment}
"""
    score_reason_prompt = f"""### Score Reason
Also, you have given a score to the current state, your reason is:
{score_reason}
"""
    sim_reflection_instruction_prompt = f"""### Reasoning Process
Reflect on the "Executed Action" you've taken. Is the action you've just executed effective in achieving the "Main Task"? 
Focus on the "Action Intent Fulfillment" in the "Observation". 
    - If not fulfilled, focus on the "Action Intent", describe what should have been done differently after the key "sim_reflection".
    - If the "Action Intent" is fulfilled, what makes it effective? Or is there any improvement that could be made? Output your reasoning after the key "sim_reflection".
"""
    if "necessary actions" in expectation.lower():
        sim_reflection_instruction_prompt += """    - Is the "Executed Action" one of the "Necessary Actions" in the "Expectation"? Was it supposed to be executed when in the same situation?
"""
    sim_reflection_instruction_prompt += """This self-reflection is crucial for identifying and learning from missteps or inaccuracies in your approach, thereby informing better decision-making when facing similar situations in the future.
"""
    format_instructions_prompt = """Format your response in JSON format, including the following key:
{
    "sim_reflection": a string that records your self-reflection on the former "Executed Action", what could be done differently, or why the current state meets the "Expectation", within 50 words.
}
"""

    prompt = gen_basic_prompt("", domain, function_description=function_description)
    prompt += observation_prompt.format(actree=actree)
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += expectation_prompt
    prompt += src_action_prompt
    prompt += observation_description_prompt
    prompt += score_reason_prompt
    prompt += sim_reflection_instruction_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt
    return prompt

# ==================== terminal_reflection ====================
def gen_terminal_reflection_prompt(task_content, domain, actree, executed_actions, ScratchPad_Info, former_node_reflections, score_reason):
    """
    works for a terminal node, generate a reflection that could be used for the whole search process.
    """
    logging.info("===== Currentprompt: gen_terminal_reflection_prompt =====")
    terminal_reflection_instruction_prompt = f"""Now, you have decided to stop and have given a score to the current state.

### Score Reason
Your reason for such a score is:
{score_reason}

### Former Node Reflections
Here are some reflections on the former states:
{former_node_reflections}

Reflect on the previous "Executed Actions" you've taken, as well as their "Former Node Reflections". Is the plan and all the actions you've taken effective in achieving the task goal?
Assess:
    - Are the "Executed Actions" effective and reasonable towards achieving the task's goal?
    - If not, from which action is the deviation from the right path?

This self-reflection is crucial for identifying and learning from missteps or inaccuracies in your approach, thereby informing better decision-making in future tasks.

Focus your reflection on the entire plan and the execution's context, offering insights that could inform future strategies and improve decision-making. 

"""
    format_instructions_prompt = """Format your response in JSON format, including the following key:
{
    "terminal_reflection": a string that records your self-reflection on the former actions.
}

"""

    prompt = gen_basic_prompt(task_content, domain)
    prompt += observation_prompt.format(actree=actree)
    if executed_actions.strip() != "":
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
    if ScratchPad_Info.strip() != "":
        prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
    
    prompt += terminal_reflection_instruction_prompt
    prompt += format_instructions_prompt
    return prompt


