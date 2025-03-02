import logging
from .prompts import *

# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with reviewing the current webpage and inspecting the actions taken by the other agents. Then you should evaluate the effectiveness according to the given criteria. \n"""
AGENT_ROLE = "appraiser"

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


# ==================== node_evaluation ====================
def gen_node_evaluation_prompt(task_content, domain, actree, executed_actions, ScratchPad_Info, obs_des_n_fulfillment, expectation):
    logging.info("===== Currentprompt: gen_node_evaluation_prompt =====")
    
    # special prompts
    function_description = """You need to evaluate the current state of the webpage to determine the effectiveness of the executed action and its potential to achieve the final goal. \n"""
    criteria_prompt = f"""### Criteria
Following is the state that should be achieved after correctly completing the task. It will be used to determine whether the task has been successfully completed.
{expectation}
"""
    observation_description_prompt = f"""### Observation Description
Here is a description generated as a summary of the current state of the webpage and how the last action has fulfilled the intent:
{obs_des_n_fulfillment}
"""
    node_evaluation_instruction_prompt = """### Evaluation
Now, assess the current web page state in terms of the effectiveness of the "Executed Action" and its potential to achieve the final goal. Your evaluation should consider the following:
**executed_action_effectiveness**: Evaluate the effectiveness of the "Executed Action", take the "Action Intent Fulfillment" into account. 
    - Was it the most efficient and logical step towards the goal? 
    - Does it have any negative effects that could lead to a wrong state?
"""
    if "necessary actions" in expectation.lower():
        node_evaluation_instruction_prompt += """    - Was the "Executed Action" one of the "Necessary Actions" in the "Criteria"?\n"""
    node_evaluation_instruction_prompt += """Evaluate the "Executed Action" and its effectiveness. Output your evaluation after the key "executed_action_effectiveness".
**executed_action_score**: Conclude your evaluation of "Executed Action" with a score reflecting its effectiveness. Use a scale from 0 (ineffective) to 10 (highly effective).
**executed Action Effectiveness Guidelines**:
    - **0-3 (Ineffective)**: The action taken was absolutely irrelevant to the goal, or even counterproductive.
    - **4-6 (Moderately Effective)**: The action taken was somewhat relevant and necessary but could be improved.
    - **7-9 (Effective)**: The action taken was mostly necessary, relevant, and efficient, significantly contributing to reaching the goal.
    - **10 (Highly Effective)**: The action taken was exactly the right path to the goal.
Analyze the executed action individually and evaluate its effectiveness. Then output your score after the key "executed_action_score".

**future_promise**: Assess the potential for the current state ("Observation") to lead to the desired goal, how likely it is to result in success? 
    - Does it already meets the "Criteria" of the subtask?
    - Are there any elements or links in the "Observation" that could lead to the final state or answer?
Output your evaluation after the key "future_promise".
**future_promise_score**: Conclude your evaluation of the future promise with a score reflecting the potential for success. Use a scale from 0 (impossible) to 10 (certain).
**Future Promise Guidelines**:
    - **0-3 (Unlikely)**: The current state seems far from reaching the goal. There are significant obstacles and little to no evidence of progress.
    - **4-6 (Possible)**: There's some potential, but the path to success isn't clear. Moderate progress has been made, but significant uncertainties exist.
    - **7-9 (Likely)**: It appears probable that the current state will lead to the goal. Most necessary conditions are met, and progress is evident.
    - **10 (Very Promising)**: The current state highly indicates achieving the desired goal. Or the goal has already been achieved.
Analyze the current state and evaluate its potential to lead to the goal. Then output your score after the key "future_promise_score".

"""
    format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "reasoning_process": string of your detailed reasoning before evaluation and the score;
    "executed_action_effectiveness": a single string describing the effectiveness of "Executed Action", including analysis of positive or maybe negative effects;
    "executed_action_score": number assigned from 0 to 10;
    "future_promise": string describing the potential for the current state to achieve the goal;
    "future_promise_score": number assigned from 0 to 10;
}
"""

    prompt = gen_basic_prompt("", domain, function_description=function_description)
    prompt += observation_prompt.format(actree=actree)
    prompt += observation_description_prompt
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += criteria_prompt
    if executed_actions.strip() != "":
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
    prompt += node_evaluation_instruction_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt
    return prompt  


# ==================== terminal_evaluation for terminal nodes ====================
def gen_terminal_evaluation_prompt(task_content, domain, expectation, actree, observation_description, executed_actions,):
    """
    We have decided to stop the search at the current process, and need to evaluate it.
    Input should be as less as possible(to be more subjective): task_content(not used), domain, actree, executed_actions;
    Evaluation prompt no longer emphasizes the "promising", but the current completion of the task.
    """
    logging.info("===== Currentprompt: gen_terminal_evaluation_prompt =====")
    
    # special prompts
    criteria_prompt = f"""### Criteria
Here is the criteria for finishing the task:
{expectation}

You should judge how the current task is completed based on the criteria above.

"""
    terminal_evaluation_instruction_prompt = f"""Now you have decided to stop the current search. It's time to assess how much of the task has been completed.

### Observation Description
The observation after executing the last action is:
{observation_description}
    
**reasoning_process**:
After that, make your assessment based on the "Observation" and the "Observation Description". Refrain from offering any recommendations.
Consider the following aspects:
    1. First, you must fully observe "Observation", also focus on the "Observation Description". Does the information of them meet the "Criteria"? Are you in the right page? Note that most of time, the main body of the webpage is the most important part. Don't be misled by the sidebar or the top bar.
    2. Second, if the "Criteria" requires some actions to be executed, are all the necessary actions executed?
    3. Third, if the "Criteria" requires an answer, and the "Observation" contains it, extract the corresponding information from the "Observation" and compare it with the "Criteria".
    
**Score**: Conclude your evaluation with a score reflecting the completion of the task. Here's a guide to help you assign a score, scaled from absolutely not(0) to perfectly(10):
    - (Inadequate): Barely addresses the criteria with little to no relevant information or excuted actions.
    - (Partially Satisfactory): Partially meets the criteria with some relevant information, but significant elements are missing.
    - (Satisfactory): Meets the criteria with relevant information, some necessary actions have been executed. The required information of a final answer can be found in the observation.
    - (Excellent): Fully meets the criteria with all relevant information and necessary actions executed. The required information is explicitly presented in the observation.

ATTENTION:
The current state is the terminal state, and no further actions will be executed. 
The information above is all you have, with no further additional details.
Therefore, do not make assumptions or further inferences(such as "it might be possible" to complete). Judge only based on the information currently available.

"""
    format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "reasoning_process": string of your detailed reasoning of evaluation;
    "score": a number measures you've assigned based on your evaluation, assigned from 0 to 10;
}
"""
    prompt = gen_basic_prompt("", domain) # currently not using task_content
    prompt += observation_prompt.format(actree=actree)
    if executed_actions.strip() != "":
        prompt += executed_action_prompt.format(executed_actions=executed_actions)
    prompt += criteria_prompt
    prompt += terminal_evaluation_instruction_prompt
    prompt += format_instructions_prompt
    return prompt

# ==================== terminal_comparison ====================
def gen_terminal_comparison_prompt(task_content, domain, expectation, node_info_1, node_info_2,):
    """
    compare two terminal nodes, and choose the better one.
    node_info is [actree, observation_description, executed_actions]
    """
    logging.info("===== Currentprompt: gen_terminal_comparison_prompt =====")
    
    # special prompts
    criteria_prompt = f"""### Criteria
Here is the criteria for finishing the task:
{expectation}

You should judge how the current task is completed based on the criteria above.

"""
    # construct info for a single node
    node_info_prompt = """### Observation
{actree}

### Observation Description
{observation_description}

### Executed Actions
{executed_actions}

"""
    node_info_1_prompt = """========== Info for Node 1 ==========\n"""
    node_info_1_prompt += node_info_prompt.format(actree=node_info_1[0], observation_description=node_info_1[1], executed_actions=node_info_1[2])
    node_info_2_prompt = """========== Info for Node 2 ==========\n"""
    node_info_2_prompt += node_info_prompt.format(actree=node_info_2[0], observation_description=node_info_2[1], executed_actions=node_info_2[2])

    actree_explanation_prompt = """### Actree Explanation
In "Observation", each line details a web element with its attributes, formatted for clarity:
    - `[element_id]`: A unique identifier for the element, represented as an integer. This ID is used to specify actions targeting the element.
    - `element_type`: Describes the element's role (e.g., 'button', 'link', 'StaticText', 'textbox'). Buttons are typically clickable; links navigate to other pages; StaticText elements are static and cannot be interacted with; textboxes allow text input.
    - `element_text`: Displays the visible text associated with the element. This text is usually static but can be dynamic for textboxes, reflecting their current value.

[Indentation] The structure also incorporates indentation to show parent-child relationships among elements. An indented element is considered a child of the directly preceding less-indented element, illustrating the hierarchy of the webpage's content.
[In-page Tabs] The page may contain multiple categorized tabs, but only the tab with "focused: True" is in the open state, indicating that you are currently under this tab.
[Dropdown Menus] Elements with "hasPopup: menu|listbox" are dropdown menus. The "expanded: True|False" attribute indicates whether the dropdown menu is open or closed, followed by its child elements.

""" 

    terminal_comparison_instruction_prompt = """### Reasoning Process
You are now comparing two terminal nodes, each representing a different state of the web page. Which of the two nodes is more likely to fulfill the "Criteria" and complete the subtask? The information for each node comes as a combination of "Observation", "Observation Description", and "Executed Actions".
Consider the following aspects:
    1. First, you must fully observe "Observation", also focus on the "Observation Description". Does the information of them meet the "Criteria"? Are you in the right page? Note that most of time, the main body of the webpage is the most important part. Don't be misled by the sidebar or the top bar.
    2. Second, if the "Criteria" requires some actions to be executed, are all the necessary actions executed?
Make assessment for each node based on the above aspects. Then compare the two nodes and decide which one is more likely to fulfill the "Criteria" and complete the subtask.

ATTENTION:
The current state is the terminal state, and no further actions will be executed. 
The information above is all you have, with no further additional details.
Therefore, do not make assumptions or further inferences(such as "it might be possible" to complete). Judge only based on the information currently available.

"""
    format_instructions_prompt = """Format your response in JSON format, including the following keys:
{
    "analysis_of_node_1": string of your detailed reasoning of "Observation", "Observation Description", and "Executed Actions" for the first node;
    "analysis_of_node_2": string of your detailed reasoning of "Observation", "Observation Description", and "Executed Actions" for the second node;
    "reasoning_process": string of your comparison between the two nodes, which one better meets the "Criteria";
    "better_node_idx": integer indicating the better node, 1 or 2.
}
"""
    prompt = gen_basic_prompt("", domain) # currently not using task_content
    prompt += node_info_1_prompt
    prompt += node_info_2_prompt
    prompt += actree_explanation_prompt
    prompt += criteria_prompt
    prompt += terminal_comparison_instruction_prompt
    prompt += format_instructions_prompt
    return prompt   
