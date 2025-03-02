import logging
from .prompts import *
import yaml


# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with planning for a web exploration task. \n"""
AGENT_ROLE = "planner"

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

# ==================== which domain ====================
def gen_which_domain_prompt(task_content):
    """
    Generate the prompt for asking which domain the task belongs to.
    """
    logging.info("===== Currentprompt: gen_which_domain_prompt =====")
    # special prompts
    which_domain_prompt = f"""### Main Task
You are an AI agent currently on a web page. You should accomplish some tasks. The current task is: 
{task_content} 

### Possible Domains
Which domain does this task belong to? Choose from the following domains and output only the domain name: 
shopping, shopping_admin, gitlab, reddit, map. 

Format your output in Json, including:
{{
    "domain": string of the domain name, choosing from: shopping, shopping_admin, gitlab, reddit, map.
}}
"""

    return which_domain_prompt

# ==================== task expectation (evaluation method prediction) ====================
def gen_task_expectation_prompt(domain, task_content, examples):
    """
    Predict the evaluation method for the task.
    """
    logging.info("===== Currentprompt: gen_task_expectation_prompt =====")
    # special prompts
    function_description = """Given a task, you should predict what actions should be executed or what information should be displayed to finish the task. \n"""

    task_expectation_prompt = f"""### Reasoning Process
After exploration and some actions, you will be evaluated in following aspects:
    1. you arrive at the target page, where the page displays the key information(which can be found in the target page);
    2. (optional) you provide the final answer, some tasks need a final answer, some don't. If there is a clear "answer requirement", then the task might probably need a final answer.
Please analyze the "Current Main Task" and output your expectation for finishing the task regarding the three aspects.
An expectation should be the following:
    - **Target Page Description**: What the target page should be like or the page should display [key] information.
    - **Answer Requirements**: First judge whether the task needs a final answer, then describe the answer if needed, concise or detailed, and what kind of the answer should be.
        When the task involves performing a specific action without needing to report the result, a final answer is not needed. For example, inviting collaborators, assigning issues, or setting due dates.
        When the task explicitly requires providing specific information or results, a final answer is needed. For example, listing the names of top contributors and their commit counts, confirming the status of some items.
You will use the expectation to judge whether the task is finished or not.

### Current Main Task
{task_content}

"""
    format_instructions_prompt = """Format your response in Json, including:
{{
    "reasoning": string of your reasoning process;
    "target_page_description": string describing the target page or the page should display [key] information;
    "need_answer": boolean value indicating whether the task needs a final answer;
    "answer_requirements": string describing the answer if needed, and what kind of the answer should be, yes/no, specific information, or both, etc.
}}
"""
    prompt = gen_basic_prompt("", domain, function_description=function_description)
    prompt += task_expectation_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt

# ==================== SubTask expectation generation ====================
def gen_subtask_expectation_prompt(task_content:str, task_domain, observation_description:str):
    """
    Generate the prompt for generating subtask expectation.
    the subtask_content is already given, generate a expecation.
    """
    logging.info("===== Currentprompt: gen_subtask_expectation_prompt =====")

    file_path = os.path.join(os.path.dirname(__file__), "demos.yaml")
    with open(file_path, "r") as f:
        demos = yaml.load(f, Loader=yaml.FullLoader)
    subtask_expectation_examples = demos[task_domain]

    examples_prompt = f"""### Examples
Here are some examples of decomposing the task into subtasks, where you can see expecations for each subtask. Do not confuse the examples with the current task:
{subtask_expectation_examples}

"""
    subtask_expectation_prompt = f"""### Observation Description
The description of the current page is:
{observation_description}
This indicates the environment you are in, which is crucial for planning the next steps.
    
### Reasoning Process
Output your expectation of the generated subtask. An expectation should be one or both of the following:
    - **target_page_description**: What the target page should be like or the page should display [key] information. This could be either general information or very specific information. Output general or specific or both kinds of information.
    - **necessary_actions**: The executed actions should include [key] action. Only include the necessary actions that must be conducted, since some actions' effects may not be visible on the page. It/They must be able to interact with the page, avoiding actions like “check,” “locate,” or “identify” which do not impact the environment.
Refer to "Examples" for more details. This expectation will be used as the criterion for judging whether the subtask is finished or not.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "target_page_description": string describing the target page or the page should display [key] information;
    "necessary_actions": string describing the necessary actions that must be conducted;
}
"""
    prompt = basic_prompt_without_info
    prompt += gen_prior_knowledge_prompt(task_domain)
    prompt += examples_prompt
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += subtask_expectation_prompt
    prompt += format_instructions_prompt

    return prompt

# ==================== gen_is_answer_needed_prompt, judge whether the task needs a specific answer ====================
def gen_is_answer_needed_prompt(task_content, domain, expectation):
    """
    let agent judge whether the task needs a specific answer. And the answer should be detailed or not.
    """
    logging.info("===== Currentprompt: gen_is_answer_needed_prompt =====")
    # special prompts
    expectation_prompt = f"""### Expectation
Here is your expectation for finishing the "Main Task":
{expectation}

It may already contain some thoughts about the demand for the answer for the "Main Task".

"""
    final_answer_prompt = """### Reasoning Process
**Is answer needed?**
Does the "Main Task" necessitate a concluding answer?
    - If the task aims to obtain specific information, then it requires an answer. 
        Examples: "Get the driving time from CMU to airport" needs a specific time; "Tell me the name of the user who made the most commits to the repository" needs a name; "Check if the latest created issue with 'bug' label is closed" needs a yes/no answer for status.
    - If the task aims to perform actions or display information, then it doesn't need a final answer. 
        Examples: "Invite a friend to the 'awesome-robotics' repository" only needs to perform the actions; "Check out the all the issues in the repository" only needs to display the issues; "Create a new issue in the repository" only needs to create the issue.

Indicate your decision with True or False assigned to `need_answer`.

**Answer Requirements**:
If `need_answer` is True.
    1. Should the answer be concise or detailed? For example, a direct command is needed for 'Give me the command to clone a repository,' whereas a question like 'Compare the time of driving and walking' demands a more elaborate answer detailing both the mode of transport and the respective durations.
    2. What kind of answer is needed? For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name.
Output your decision in a short sentence after the key "answer_requirements".

"""

    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your thinking process about whether the task needs a final answer, refer to examples;
    "need_answer": boolean value indicating whether the task needs a final answer;
    "answer_requirements": string describing the answer you need, what kind of the answer should be;
}
"""

    prompt = gen_basic_prompt(task_content, domain)
    prompt += expectation_prompt    
    prompt += final_answer_prompt
    prompt += format_instructions_prompt
    return prompt

# ==================== gen_is_answer_needed_by_subtask_prompt ====================
def gen_is_answer_needed_by_subtask_prompt(task_content, domain, expectation,):
    """
    let agent judge whether the task needs a specific answer. And the answer should be detailed or not.
    """
    logging.info("===== Currentprompt: gen_is_answer_needed_by_subtask_prompt =====")
    # special prompts
    expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}

It may already contain some thoughts about the demand for the answer for the "Main Task".

"""
    final_answer_prompt = """Does the "Main Task" necessitate a concluding answer?
- Tasks that need an answer are to obtain specific information or instructions. 
- Tasks that don't need an answer are to perform actions or display information. 
For example, "Get the zip-code of the address" needs a specific zip-code, but "Navigate to the homepage of the repository" only needs to perform the action.

Indicate your decision with True or False assigned to `need_answer`.

- If `need_answer` is True, what kind of answer is needed?
For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name. Sometimes the answer needs to be detailed, sometimes it can be simple. Describe the answer you need in a short sentence after the key "answer_requirements".

Output your analysis after the key "thinking_process", and other info after the corresponding keys.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your thinking process about whether the task needs a final answer, what kind of the answer should be;
    "need_answer": boolean value indicating whether the task needs a final answer;
    "answer_requirements": string describing the answer you need, what kind of the answer should be;
}
"""
    prompt = gen_basic_prompt(task_content, domain)
    prompt += expectation_prompt
    prompt += final_answer_prompt
    prompt += format_instructions_prompt
    return prompt

# ==================== gen_answer_requirements_prompt ====================
def gen_answer_requirements_prompt(task_content, domain, expectation,):
    """
    is_answer_needed is True, then let agent judge what kind of answer is needed.
    """
    logging.info("===== Currentprompt: gen_answer_requirements_prompt =====")
    # special prompts
    expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}

It may already contain some thoughts about the demand for the answer for the "Main Task".

"""
    answer_requirements_instruction = """You need an answer for the "Main Task". What kind of answer is needed?
For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name. Sometimes the answer needs to be detailed, sometimes it can be simple. Describe the answer you need in a short sentence after the key "answer_requirements".

Output your analysis after the key "thinking_process", and other info after the corresponding keys.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your thinking process about what kind of the answer for the "Main Task" should be, 
    "answer_requirements": string describing the answer you need, what kind of the answer should be;
}
"""
    prompt = gen_basic_prompt(task_content, domain)
    prompt += expectation_prompt
    prompt += answer_requirements_instruction
    prompt += format_instructions_prompt
    return prompt

# ==================== gen_is_only_info_extraction_prompt ====================
def gen_is_only_info_extraction_prompt(task_content, domain, expectation, actree):
    """
    let agent judge whether the task only needs to extract information without interacting with the webpage.
    """
    logging.info("===== Currentprompt: gen_is_only_info_extraction_prompt =====")
    # special prompts
    expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}

It may already contain some thoughts about the demand for the answer for the "Main Task".
    
"""
    only_info_extraction_prompt = """Now you need to judge whether the task only needs to extract information without interacting with the webpage.
- Think about the "Main Task" and the "Expectation", what kind of information is needed to finish the task? 
- Observe the current "Observation" and analyze the current page. What kind of page are you on, and what does the page look like?
- Is the information required for the "Main Task" and its "Expectation" already displayed on the current "Observation" page? If so, you may set the "only_info_extraction" to True.
- Some tasks must be "only_info_extraction", as the interaction with the webpage may change the state of some specific items(irreversible), or lead to information loss. If you think the current "Main Task" has this possibility, you should consider setting "only_info_extraction" to True.

Output your analysis after the key "thinking_process", and the decision after the key "only_info_extraction".

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your thinking process about whether the task only needs to extract information without interacting with the webpage;
    "only_info_extraction": boolean value indicating whether the task only needs to extract information;
}
"""
    prompt = gen_basic_prompt("", domain)
    prompt += observation_prompt.format(actree=actree)
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += expectation_prompt
    prompt += only_info_extraction_prompt
    prompt += format_instructions_prompt
    return prompt

# ==================== whether decompose ====================
def gen_whether_decompose_prompt(task_domain, task_content, task_expectation, observation_description,):
    """
    Generate the prompt for whether task needs to be decomposed.
    """
    logging.info("===== Currentprompt: gen_whether_decompose_prompt =====")
    # special prompts
    function_description = """Given a task, you should judge whether the task needs to be decomposed. \n"""

    whether_decompose_prompt = gen_basic_prompt("", task_domain, function_description=function_description)
    whether_decompose_prompt += f"""
### Main Task
Now the task is:
{task_content}

### Expectation
Here is your expectation for finishing the task:
{task_expectation}
### Observation Description
The description of the current page is:
{observation_description}
This indicates the environment you are in, which is crucial for planning the next steps.

### Reasoning Process
**Decomposition Requirement**: Do you think the "Main Task" needs to be decomposed? 
If the “Main Task” requires multiple steps or subtasks to be performed to achieve the final goal, then it needs to be decomposed.

Format your response in Json, including:
{{ 
    "decomposition_requirement": string anaylsing the necessity of decomposition of the "Main Task", if there exists multi subtasks, then it needs to be decomposed;
    "task_needs_decomposition": boolean value whether the task needs to be decomposed;
}}
"""
    return whether_decompose_prompt

# ==================== generate plan ====================
def gen_generate_plan_prompt(task_domain, task_content, task_expectation, observation_description, actree=""):
    """
    Generate the prompt for whether task needs to be decomposed.
    """
    logging.info("===== Currentprompt: gen_generate_plan_prompt generate plan! =====")
    # special prompts
    function_description = """Given a task, you should analyze it and decompose it into subtasks to generate a plan. \n"""

    file_path = os.path.join(os.path.dirname(__file__), "demos.yaml")
    with open(file_path, "r") as f:
        demos = yaml.load(f, Loader=yaml.FullLoader)
    subtasks_examples = demos[task_domain]

    examples_prompt = f"""### START of Examples
Here are some examples of decomposing the task into subtasks, do not confuse the examples with the "Current Main Task". Pay attention to the granularity of each subtask; when generating the plan, ensure that the subtasks are at the same level of detail.
{subtasks_examples}
### END of Examples

"""
    current_maintask_prompt = f"""### Current Main Task
This is the task you need to decompose and create a plan for:
{task_content}

"""
    generate_plan_prompt = ""
    from WebPilot.model import WITH_EXPECTATION_FLAG
    if WITH_EXPECTATION_FLAG:
        generate_plan_prompt += f"""### Main Task Expectation
Here is your expectation for finishing the task:
{task_expectation}

"""
    generate_plan_prompt += f"""### Observation Description
The description of the current page is:
{observation_description}
This indicates the environment you are in, which is crucial for planning the next steps.

### Reasoning Process
For complex tasks, it’s crucial to decompose the task into the smallest, manageable subtasks. Each subtask should focus on accomplishing a specific, individual goal to ensure clarity and precision in execution.
Keep the following aspects in mind:
1. Make sure the plan contains all necessary subtasks. Every aspect of the "Current Main Task" should correspond to a subtask. Try to cover every word in the "Current Main Task"."""
    if WITH_EXPECTATION_FLAG:
        generate_plan_prompt += f"""
    Note that the expectations of generated subtasks should cover the whole "Main Task Expectation"."""
    generate_plan_prompt += f"""
    Provide your reasoning process for each subtask to explain why it is necessary. Don't forget the current state described in the "Observation Description". Also analyze the connection between this subtask and its previous and next subtasks. Output your reasoning after the key "reasoning".
2. Each subtask should be followed by its expectation. An expectation should be one or both of the following:
    - "target_page_description": What the target page should be like or the page should display [key] information. This can be either general or very specific information.
    - "necessary_actions": The executed actions should include [key] action. Only include the necessary actions that must be conducted, since some actions' effects may not be visible on the page. It/They must be able to interact with the page, avoiding actions like “check,” “locate,” or “identify” which do not impact the environment. Also, index the actions in the order they should be executed, even if only one action is needed.
    Refer to "Examples" for more details on how to structure and decompose tasks. However, make sure not to confuse the examples with the "Current Main Task". The examples are provided to illustrate the process of task decomposition, while the "Current Main Task" is the specific task you need to address. The expectation provided will be used to judge whether each subtask is finished.
3. Each subtask needs to be evaluated to determine if it requires an answer. If the subtask's aim is to obtain specific information, then it requires an answer.

"""
    caution_prompt = """**IMPORTANT**
When generating a plan, ALWAYS check the following:
1. Compliance with Rules: EACH subtasks in the generated plan MUST comply with ALL relevant rules outlined in the “Prior Knowledge” to ensure full compliance. When generating the plan, first identify the relevant rules and then incorporate them accordingly.
2. Interaction with Webpage: EACH subtasks in the generated plan MUST involve a specific interaction with the webpage to ensure they are actionable and concrete. Do NOT generate subtasks that start with words like “locate,” “identify,” or “observe,” as these imply observation without interaction. Subtasks must require direct interaction with the webpage.
3. Granularity: EACH subtasks in the generated plan MUST be the smallest possible unit. If a subtask is at a higher level of detail compared to the subtasks in the “Examples”, decompose it further until each subtask focuses on accomplishing a specific, individual goal.

Output your plan of the "Current Main Task", making sure to follow the guidelines provided above.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "plan": list of the plan of the "Current Main Task", with each subtask as dict including the key "reasoning", "subtask", "target_page_description", "necessary_actions", and "need_answer";
}
"""

    prompt = basic_prompt_without_info
    prompt += function_description
    prompt += examples_prompt
    if actree.strip() != "":
        prompt += observation_prompt.format(actree=actree)
    prompt += current_maintask_prompt
    prompt += generate_plan_prompt
    prior_knowledge_prompt = gen_prior_knowledge_prompt(task_domain)
    prior_knowledge_prompt = prior_knowledge_prompt.replace("Main Task", "Current Main Task")
    prompt += prior_knowledge_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += caution_prompt
    prompt += format_instructions_prompt

    return prompt

# ==================== whether update plan needed ====================
def gen_whether_update_plan_prompt(task_content, task_domain, expectation, actree, observation_description, plan, finished_subtasks, ScratchPad_Info, 
                           not_stop_reason, sim_subtask, sim_subtask_reflection,
                           ):
    """
    before update plan, ask whether the agent needs to update the plan.
    """
    logging.info("===== Currentprompt: gen_whether_update_plan_prompt =====")

    # special prompts
    # get examples
    file_path = os.path.join(os.path.dirname(__file__), "demos.yaml")
    with open(file_path, "r") as f:
        demos = yaml.load(f, Loader=yaml.FullLoader)
    subtasks_examples = demos[task_domain]
    examples_prompt = f"""### START of Examples
Here are some examples of decomposing the task into subtasks, do not confuse the examples with the "Current Main Task". Pay attention to the granularity of each subtask; when generating the plan, ensure that the subtasks are at the same level of detail.
{subtasks_examples}
### END of Examples

"""
    current_maintask_prompt = f"""### Current Main Task
This is the task you need to judge whether the plan needs to be updated:
{task_content}

"""
    observation_description_prompt = f"""### Observation Description
The description of the current page is:
{observation_description}

"""
    if plan.strip() == "":
        plan = "Currently, the plan is empty, meaning that you may have finished the last subtask."
    rough_plan_prompt = f"""### Rough Plan
{plan}
Note: The plan may contain speculative or 'hallucinated' subtasks, due to limited information available at the time of planning.

"""
    finished_subtasks_prompt = f"""### Finished Subtasks
Here are the finished subtasks conducted by yourself, note that it is not guaranteed that all the subtasks are completed, the completion status are also given:
{finished_subtasks}
"""
    not_stop_reason_prompt = f"""### Reason for not stopping
Focus on the last subtask in the "Finished Subtasks", since it leads you to the current situation. You have analyzed the reason for not stopping the "Current Main Task":
{not_stop_reason}

"""
    if sim_subtask.strip() != "":
        last_subtask_reflection_prompt = f"""### Reflection on the Last Subtask
Actually, you have tried a subtask in the same situation before, but you didn't complete it. You have also made a reflection on it.
**Incomplete Subtask**: {sim_subtask}
**Incomplete Subtask Reflection**: {sim_subtask_reflection}
It may also provide some insights for you to avoid the same mistake or to improve the plan.

"""    
    whether_update_plan_prompt = """### Reasoning Process
For complex tasks, it's important to break down the task into simpler, manageable subtasks. Each subtask should aim at achieving a specific, singular goal. Each subtask must involve interaction with the webpage to ensure it is actionable and specific.
However, due to a lack of detailed understanding of the environment before starting the task, the "Rough Plan" may not be suitable. Now judge whether the plan needs to be updated.
    1. Analyze the current "Observation" and "Observation Description" of the current page. What could be the next step to achieve the "Current Main Task" ? Does it align with the subtasks in the "Rough Plan"?
    2. Consider the other subtasks in the "Rough Plan", do they omit any necessary steps or are the redunant? 
        - Some necessary steps may be omitted due to the simple description of the task. For example: Words similar to "latest" or "newest" may need a special subtask to sort the current items. Are there any missing subtasks in both "Finished Subtasks" and "Rough Plan"?
        - Some subtasks may be redundant and can be combined. For example: "Identify the link to merge request" and "Click the link to merge request" can be combined into one subtask. Are there any redundant subtasks in both "Finished Subtasks" and "Rough Plan"?
    3. Consider the "Reason for not stopping", is there any new information that could help you to update the plan?
    4. Refer to "Examples" for more details on how to structure and decompose tasks. However, make sure not to confuse the examples with the "Current Main Task". The examples are provided to illustrate the process of task decomposition, while the "Current Main Task" is the specific task you need to address. The expectation provided will be used to judge whether each subtask is finished.
"""
    if sim_subtask.strip() != "":
        whether_update_plan_prompt += """   5. Is there any useful information in the "Reflection on the Last Subtask"?\n"""
    if "Answer Requirements" in expectation:
        whether_update_plan_prompt += """   6. Consider the "Answer Requirements" in the expectation and the "ScratchPad Info" if there is any. What else information is needed?\n"""
    whether_update_plan_prompt += """Output your decision after the key "need_udpate_plan".
If plan needs to be updated, how will you do that? Output your advice on the improvement of the plan after the key "update_plan_advices".

"""
    prior_knowledge_prompt = gen_prior_knowledge_prompt(task_domain)
    prior_knowledge_prompt = prior_knowledge_prompt.replace("Main Task", "Current Main Task")
    prior_knowledge_prompt += "Consider these knowledge for your decision. \n"
    format_instructions_prompt = """Format your response in Json, including:
{   
    "current_situation": string of the current situation;
    "rough_plan_necessity": string analyzing the necessity of each subtask in the "Rough Plan";
    "possible_omission": string analyzing the possible omission of subtasks in both the "Finished Subtasks" and the "Rough Plan", output None if not needed;
    "possible_redundancy": string analyzing the possible redundancy of subtasks in both the "Finished Subtasks" and the "Rough Plan", output None if not needed;
    "reasoning_process": string of your reasoning process whether the plan needs to be updated;
    "need_update_plan": boolean value indicating whether the plan needs to be updated;
    "update_plan_advices": string describing how the plan could be updated.
}
"""
    prompt = basic_prompt_without_info
    prompt += examples_prompt
    prompt += observation_prompt.format(actree=actree)
    prompt += observation_description_prompt
    prompt += current_maintask_prompt
    prompt += rough_plan_prompt
    # if ScratchPad_Info.strip() != "":
    #     prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
    if finished_subtasks.strip() != "":
        prompt += finished_subtasks_prompt
    if not_stop_reason.strip() != "":
        prompt += not_stop_reason_prompt
    if sim_subtask.strip() != "":
        prompt += last_subtask_reflection_prompt
    prompt += whether_update_plan_prompt
    prompt += prior_knowledge_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt

# ==================== udpate plan ====================
def gen_update_plan_prompt(task_content, task_domain, expectation, actree, observation_description, plan, finished_subtasks, ScratchPad_Info, 
                           not_stop_reason, sim_subtask, sim_subtask_reflection, update_plan_advices,
                           ):
    """
    if agent decide not to stop when asking about the stop decision, then update the plan.
    Works as a external refinment of the plan.
    """
    logging.info("===== Currentprompt: gen_update_plan_prompt =====")

    # special prompts
    # get examples
    file_path = os.path.join(os.path.dirname(__file__), "demos.yaml")
    with open(file_path, "r") as f:
        demos = yaml.load(f, Loader=yaml.FullLoader)
    subtasks_examples = demos[task_domain]
    examples_prompt = f"""### START of Examples
Here are some examples of decomposing the task into subtasks, do not confuse the examples with the "Current Main Task". Pay attention to the granularity of each subtask; when generating the plan, ensure that the subtasks are at the same level of detail.
{subtasks_examples}
### END of Examples

"""
    current_maintask_prompt = f"""### Current Main Task
This is the task you need to decompose and create a plan for:
{task_content}

"""
    observation_description_prompt = f"""### Observation Description
The description of the current page is:
{observation_description}

"""
    if plan.strip() == "":
        plan = "Currently, the plan is empty, meaning that you may have finished the last subtask."
    rough_plan_prompt = f"""### Rough Plan
{plan}
Note: The plan may contain speculative or 'hallucinated' subtasks, due to limited information available at the time of planning.

"""
    finished_subtasks_prompt = f"""### Finished Subtasks
Here are the finished subtasks conducted by yourself, note that it is not guaranteed that all the subtasks are completed, the completion status are also given:
{finished_subtasks}
"""
    not_stop_reason_prompt = f"""### Reason for not stopping
Focus on the last subtask, since it leads you to the current situation. You have analyzed the reason for not stopping the task:
{not_stop_reason}

Take it into consideration and decide whether to redo the last subtask or refine the plan.

"""
    if sim_subtask.strip() != "":
        last_subtask_reflection_prompt = f"""### Reflection on the Last Subtask
Actually, you have tried a subtask in the same situation before, but you didn't complete it. You have also made a reflection on it.
**Incomplete Subtask**: {sim_subtask}
**Incomplete Subtask Reflection**: {sim_subtask_reflection}

It may also provide some insights for you to avoid the same mistake or to improve the plan.

"""
    update_plan_hints = f"""### Update Plan Hints
{update_plan_advices}

""" 
    
    update_plan_instruction_prompt = f"""### Reasoning Process
For complex tasks, it's important to break down the task into simpler, manageable subtasks. Each subtask should aim at achieving a specific, singular goal. Each subtask must involve interaction with the webpage to ensure it is actionable and specific.
However, due to a lack of detailed understanding of the environment before starting the task, the "Rough Plan" may not be suitable. Therefore, update your plan now.
    1. Analyze the current "Observation" and "Observation Description" of the current page. What could be the next step to achieve the "Current Main Task"?
    2. Consider the "Rough Plan" and the "Finished Subtasks", are they all necessary? Are there any missing subtasks? Are there any redundant subtasks?
    3. Think about the "ScratchPad Information", what information is critical for the task? What information is missing?
    4. Consider the "Reason for not stopping", is there any new information that could help you to update the plan?
    5. Consider the advices from the "Update Plan Hints".
"""
    if sim_subtask.strip() != "":
        update_plan_instruction_prompt += f"""  6. Is there any useful information in the "Reflection on the Last Subtask"? Consider:
        - If the "Incomplete Subtask" is unreasonable, you should refine it.
        - Is the order of the subtasks in the "Rough Plan" not correct? Also consider the "Finished Subtasks". Should the "Incomplete Subtask" be done earlier or later?
"""        
    update_plan_instruction_prompt += f"""
**Plan Generation**:
1. Make sure the plan contains all the necessary subtasks. 
    - Note that the expectations of generated subtasks should cover the whole "Main Task Expectation".
    - Each subtask should be at a low level, with only a single clear objective. The level should be similar to the "Examples".
    Provide your reasoning process for each subtask, to explain why you need to do this subtask. Also analyze the connection between this subtask and its previous and next subtasks. Output your reasoning after the key "reasoning".
2. Each subtask should be followed by its expectation. An expectation should be one or both of the following:
    - "target_page_description": What the target page should be like or the page should display [key] information. This could be either general information or very specific information. Output general or specific or both kinds of information.
    - "necessary_actions": The executed actions should include [key] action. Only include the necessary actions that must be conducted, since some actions' effects may not be visible on the page. It/They must be able to interact with the page, avoiding actions like “check,” “locate,” or “identify” which do not impact the environment.
    Refer to "Examples" for more details. This expectation will be used as the criterion for judging whether the subtask is finished or not.
3. Each subtask needs to be evaluated to determine if it requires an answer. If the subtask's aim is to obtain specific information, then it requires an answer. When solving this kind of subtask, you are not allowed to interact with the elements on the page, but you can observe the page or scroll the page to find the information. Hence, before this kind of subtask, you need plan other subtasks to get to the target page. 
    - Tasks that need an answer are to obtain specific information or instructions. 
    - Tasks that don't need an answer are to perform actions or navigate to a specific page to display information.
    Indicate your decision with True or False assigned to `need_answer`.

**Updated Plan**:
Based on the above analysis, update the plan, STARTING FROM THE CURRENT SITUATION. The first subtask in the "updated_plan" should be the next step you need to take. Each subtask must involve interaction with the webpage to ensure it is actionable and specific.

"""
    prior_knowledge_prompt = gen_prior_knowledge_prompt(task_domain)
    prior_knowledge_prompt = prior_knowledge_prompt.replace("Main Task", "Current Main Task")
    prior_knowledge_prompt += "Consider these knowledge for your decision."
    format_instructions_prompt = """Format your response in Json, including:
{   
    "current_situation": string of the current situation;
    "rough_plan_necessity": string analyzing the necessity of each subtask in the "Rough Plan";
    "possible_omission": string analyzing the possible omission of subtasks in the "Rough Plan", output None if not needed;
    "possible_redundancy": string analyzing the possible redundancy of subtasks in the "Rough Plan", output None if not needed;
    "reasoning_process": string of your reasoning process to update the plan;
    "updated_plan": list of the plan, with each subtask as dict including the key "reasoning", "subtask", "target_page_description", "necessary_actions", and "need_answer", format refers to the given "Examples";
}
"""
    
    prompt = basic_prompt_without_info
    prompt += observation_prompt.format(actree=actree)
    prompt += observation_description_prompt
    prompt += examples_prompt
    prompt += current_maintask_prompt
    prompt += rough_plan_prompt
    # if ScratchPad_Info.strip() != "":
    #     prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
    if finished_subtasks.strip() != "":
        prompt += finished_subtasks_prompt
    if not_stop_reason.strip() != "":
        prompt += not_stop_reason_prompt
    if sim_subtask.strip() != "":
        prompt += last_subtask_reflection_prompt
    prompt += update_plan_hints
    prompt += update_plan_instruction_prompt
    prompt += prior_knowledge_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt

# ==================== gen_next_subtask_to_execute_prompt ====================
class gen_next_subtask_to_execute_prompt():
    """
    analyse whether the next subtask should be executed.
    """
    expectation_prompt = """### Expectation
The expectation is the state that should be achieved after correctly completing the “Main Task”. It will be used to determine whether the “Main Task” has been successfully completed.
{expectation}
"""
    current_subtask_prompt = """### Subtask to be executed
The next subtask supposed to be executed is:
{current_subtask}
"""
    finished_subtasks_prompt = """### Finished Subtasks
Here are the finished subtasks conducted by yourself, the completion status are also given:
{finished_subtasks}
"""
    plan_prompt = """### Rough Plan
Here is the rough plan containing the subtasks that have not been executed yet:
{plan}
"""

    @staticmethod
    def redundancy(domain, maintask, expectation, current_subtask, finished_subtasks, last_subtask_actions,):
        logging.info("===== Currentprompt: gen_next_subtask_to_execute_prompt.redundancy =====")
        # special prompts
        last_subtask_actions_prompt = f"""### Last Subtask Actions
In the last subtask, you have conducted the following actions:
{last_subtask_actions}
"""
        whether_execute_next_subtask_prompt = """### Reasoning Process
To complete the "Main Task" and its "Expectation", you need to decide whether the "Subtask to be executed" should be executed next.
Is the "Subtask to be executed" redundant with the "Finished Subtasks"? 
Observe the "Last Subtask Actions", have the "Finished Subtasks" already completed the intent of the "Subtask to be executed"? Consider 2 aspects:
    1. Is the "Target Page" of the "Subtask to be executed" already achieved by the "Finished Subtasks"? Notice that the specific outcomes required for each similar but individual user or item must be achieved.
    2. If applicable, are the "Necessary Actions" of the "Subtask to be executed" already conducted by the "Finished Subtasks"? Remember, each specific user's or item's actions must be completed individually, even if the process is similar.
Output your analysis after the key "reasoning_of_redundancy" and output the decision after the key "redundant". 

"""
        format_instructions_prompt = """Format your response in Json, including:
{
    "reasoning_of_redundancy": string of your reasoning process about the redundancy of the "Subtask to be executed";
    "redundant": boolean value indicating whether the "Subtask to be executed" is redundant;
}
"""
        prompt = gen_basic_prompt("", domain)
        prompt += maintask_prompt.format(task_content=maintask)
        prompt += gen_next_subtask_to_execute_prompt.expectation_prompt.format(expectation=expectation)
        prompt += gen_next_subtask_to_execute_prompt.finished_subtasks_prompt.format(finished_subtasks=finished_subtasks)
        # prompt += gen_next_subtask_to_execute_prompt.plan_prompt.format(plan=plan)
        prompt += last_subtask_actions_prompt
        prompt += gen_next_subtask_to_execute_prompt.current_subtask_prompt.format(current_subtask=current_subtask)
        prompt += whether_execute_next_subtask_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt
    
    @staticmethod
    def decomposition(domain, maintask, expectation, current_subtask,):
        logging.info("===== Currentprompt: gen_next_subtask_to_execute_prompt.decomposition =====")
        # special prompts
        file_path = os.path.join(os.path.dirname(__file__), "demos.yaml")
        with open(file_path, "r") as f:
            demos = yaml.load(f, Loader=yaml.FullLoader)
        subtasks_examples = demos[domain]

        examples_prompt = f"""### START of Examples
Here are some examples of decomposing the task into subtasks, do not confuse the examples with the "Current Main Task". Pay attention to the granularity of each subtask; when generating the plan, ensure that the subtasks are at the same level of detail.
{subtasks_examples}
### END of Examples

"""
        whether_execute_next_subtask_prompt = """### Reasoning Process
To complete the "Main Task" and its "Expectation", you need to decide whether the "Subtask to be executed" should be executed next.
Is the "Subtask to be executed" too complex and should be further decomposed?
Refer to the "Examples" for more details on how to structure and decompose tasks. However, make sure not to confuse the examples with the "Subtask to be executed". The examples are provided to illustrate the process of task decomposition, while the "Subtask to be executed" is the specific task you need to address.
Output your analysis after the key "reasoning_of_decomposition" and output the decision after the key "need_decomposition".

"""
        format_instructions_prompt = """Format your response in Json, including:
{
    "reasoning_of_decomposition": string of your reasoning process about the necessity of decomposing the "Subtask to be executed";
    "need_decomposition": boolean value indicating whether the "Subtask to be executed" needs to be decomposed;
}
"""
        prompt = gen_basic_prompt("", domain)
        prompt += examples_prompt
        prompt += maintask_prompt.format(task_content=maintask)
        prompt += gen_next_subtask_to_execute_prompt.expectation_prompt.format(expectation=expectation)
        prompt += gen_next_subtask_to_execute_prompt.current_subtask_prompt.format(current_subtask=current_subtask)
        prompt += whether_execute_next_subtask_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt
    
    @staticmethod
    def order(domain, maintask, expectation, current_subtask, plan,):
        logging.info("===== Currentprompt: gen_next_subtask_to_execute_prompt.order =====")
        # special prompts
        whether_execute_next_subtask_prompt = """### Reasoning Process
To complete the "Main Task" and its "Expectation", you need to decide whether the "Subtask to be executed" should be executed next.
Analyse each subtask in the "Rough Plan", should it be executed before the "Subtask to be executed"? 
Output your analysis after the key "reasoning_of_order" and output the decision after the key "need_order".

"""
        format_instructions_prompt = """Format your response in Json, including:
{
    "reasoning_of_order": string of your reasoning process about the order of the "Subtask to be executed";
    "need_order": boolean value indicating whether the "Subtask to be executed" should be executed before other subtasks;
}
"""
        prompt = gen_basic_prompt("", domain)
        prompt += maintask_prompt.format(task_content=maintask)
        prompt += gen_next_subtask_to_execute_prompt.expectation_prompt.format(expectation=expectation)
        prompt += gen_next_subtask_to_execute_prompt.current_subtask_prompt.format(current_subtask=current_subtask)
        prompt += gen_next_subtask_to_execute_prompt.plan_prompt.format(plan=plan)
        prompt += whether_execute_next_subtask_prompt
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt
        return prompt
    
    @staticmethod
    def gen_right_subtask():
        """
        if the "Subtask to be executed" should be decomposed, or should be reordered, then generate the right subtask to execute.
        """
        pass


# ==================== gen_final_answer_prompt ====================
def gen_final_answer_prompt_old(task_content, domain, actree, ScratchPad_Info, expectation):
    """
    used for generating the final answer, from all the intermediate information (ScratchPad_Info), and final obervation (actree)
    """
    logging.info("===== Currentprompt: gen_final_answer_prompt =====")

    # special prompts
    expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}
"""
    final_answer_prompt = """### Reasoning Process
Now you have decided to stop the task. Generate the final answer. Consider the following aspects:
    1. According to "Expectation", what kind of a final answer is needed?
    2. Is the final answer specific information, which could be found in the "Observation"? If so, extract it.
    3. Does the final answer need multiple pieces of information? If so, organize them mainly based on the "ScratchPad Information". If there are multiple information pieces which are confilicting, you should choose the most reliable one considering its corresponding "subtask".
Output your thinking process and the final answer.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your final thinking process about the answer;
    "final_answer": string of the final answer, output None if not needed;
}
"""
    prompt = gen_basic_prompt("", domain)
    prompt += observation_prompt.format(actree=actree)
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += expectation_prompt
    if ScratchPad_Info.strip() != "":
        prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
    prompt += final_answer_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt
    return prompt

def gen_final_answer_prompt(task_content, domain, actree, generated_answer):
    """
    sometimes the generated answer does not satisfy the main task. so ask the agent to refine it.
    """
    logging.info("===== Currentprompt: gen_final_answer_prompt =====")

    # special prompts
    function_description = """Given a task, you need to verify whether the generated answer satisfies the task requirement. If not, refine the answer. \n"""

    final_answer_prompt = f"""### Reasoning Process
From the current "Observation", you have generated the answer:
{generated_answer}

Do you think the answer satisfied the "Main Task"? 
If yes, output true after the key "answer_satisfied".
If not, output false after the key "answer_satisfied" and refine the answer considering the following aspects:
    1. According to the "Main Task", what kind of a final answer is needed? Should it be a phrase, a simple number, or a detailed explanation?
    2. Information can be extracted from the current "Observation".
    3. If the answer is "N/A", adapt it to the required format, e.g. "None" for requried information not found, 0 for required number not found.
Output your answer after the key "refined_answer".

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "answer_satisfied": boolean value indicating whether the generated answer satisfies the "Main Task";
    "refined_answer": string of the refined answer, information extracted from the current "Observation";
}
"""
    prompt = gen_basic_prompt("", domain, function_description=function_description)
    prompt += observation_prompt.format(actree=actree)
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += final_answer_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt

    return prompt


# ==================== gen_subtask_final_answer ====================
def gen_subtask_final_answer_prompt(task_content, domain, expectation, actree, observation_description,):
    """
    used for generating the final answer of a subtask
    """
    logging.info("===== Currentprompt: gen_subtask_final_answer_prompt =====")

    # special prompts
    expectation_prompt = f"""### Expectation
Here is the expectation for finishing the subtask, which may include the demand for a final answer:
{expectation}

"""
    observation_description_prompt = f"""### Observation Description
Here is the observation description of the current environment generated by yourself:
{observation_description}

"""
    
    final_answer_prompt = """Now you have decided to stop the subtask. It's time to generate the final answer. Consider the following aspects:
- According to "Expectation", what kind of a final answer is needed?
- Is the final answer specific information, which could be found in the "Observation"? If so, extract it.

Output your thinking process and the final answer.

"""
    format_instructions_prompt = """Format your response in Json, including:
{
    "thinking_process": string of your final thinking process about the answer;
    "final_answer": string of the final answer;
}
"""

    prompt = gen_basic_prompt(task_content, domain)
    prompt += expectation_prompt
    prompt += observation_prompt.format(actree=actree)
    if observation_description.strip() != "":
        prompt += observation_description_prompt
    prompt += final_answer_prompt
    prompt += format_instructions_prompt
    return prompt


# ==================== gen_whether_call_info_extractor_prompt ====================
class gen_whether_call_info_extractor_prompt:
    @staticmethod
    def info_needed(task_content, task_domain, expectation, plan, finished_subtasks, ScratchPad_Info, not_stop_reason):
        logging.info("===== Currentprompt: gen_whether_call_info_extractor_prompt.info_needed =====")
        # special prompts
        maintask_expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}
"""
        if plan.strip() == "":
            plan = "Currently, the plan is empty, meaning that you may have finished the last subtask."
        rough_plan_prompt = f"""### Rough Plan
Here is the rough plan containing the subtasks that have not been executed yet:
{plan}
Note: The plan may contain speculative or 'hallucinated' subtasks, due to limited information available at the time of planning.

"""
        if finished_subtasks.strip() == "":
            finished_subtasks = """Currently, there are no finished subtasks, meaning that you have just started the task."""
        finished_subtasks_prompt = f"""### Finished Subtasks
Here are the finished subtasks conducted by yourself, note that it is not guaranteed that all the subtasks are completed, the completion status are also given:
{finished_subtasks}

"""
        empty_ScratchPad_Info = """### ScratchPad Information
Currently, there is no information in the ScratchPad. If the "Criteria" needs some answer to be stored, it is possiblly you have not found it, which means you cannot stop the task now.
"""
        not_stop_reason_prompt = f"""### Reason for not stopping
Focus on the last subtask, since it leads you to the current situation. You have analyzed the reason for not stopping the task:
{not_stop_reason}

Take it into consideration and decide whether to redo the last subtask or refine the plan.

"""    
        whether_call_info_extractor_instruction = f"""### Reasoning Process
**Information Requirements**:
What kind of information do you need? Consider the following aspects:
    1. What does the "Main Task" and "Expectation" require? Is it a single piece of information, or multiple pieces? 
    2. If it is multiple pieces, think about the "ScratchPad Information", which contains the information you have collected. What else is needed to fulfill the "Main Task" and its "Expectation"?
Describe the information, including following aspects:
    1. Information Description: a simple and clear description of this information, output after the key "info_description";
    2. Information Requirements: 
        - Should the info be concise or detailed? For example, a direct command is needed for 'Give me the command to clone a repository,' whereas a question like 'Compare the time of driving and walking' demands a more elaborate answer detailing both the mode of transport and the respective durations.
        - What kind of info is needed? For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name.
        Output the requirements after the key "info_requirements".

"""
        format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "info_description": string of simple, clear description of the information you want to extract from the current page;
    "info_requirements": string of the concrete requirements of the information, concise or detailed, what kind of info it is;
}
"""
        prompt = gen_basic_prompt("", task_domain)
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += maintask_expectation_prompt
        prompt += finished_subtasks_prompt
        prompt += rough_plan_prompt
        if ScratchPad_Info.strip() != "":
            prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
        else:
            prompt += empty_ScratchPad_Info
        prompt += not_stop_reason_prompt
        prompt += whether_call_info_extractor_instruction
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt

        return prompt
    
    @staticmethod
    def contain_info(task_content, task_domain, actree, observation_description, info_needed):
        logging.info("===== Currentprompt: gen_whether_call_info_extractor_prompt.contain_info =====")

        # special prompts
        function_description = """You should decide whether the current page contains the information needed by the "Main Task" and "Information Needed". \n"""
        observation_description_prompt = f"""### Observation Description
{observation_description}

"""
        info_needed_prompt = f"""### Information Needed
{info_needed}

"""
        whether_call_info_extractor_instruction = f"""### Reasoning Process:
Does the current "Observation" directly contain the information needed by "Main Task" and "Information Needed"?
Does the "Observatoin Description" shows that the current page is the right page to extract the information? 
Note that you can't interact with the elements on the page, i.e. you can only observe it and scroll to view it. So don't make judgments based on the links on the page. 
Output your reasoning after the key "reasoning_of_observation" why this page includes the information needed. Then output True/False after the key "observation_contains_info" to indicate whether the information is directly available in the current "Observation".

"""
        format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "reasoning_of_observation": string of your reasoning, whether the desired information is directly available in the current "Observation";
    "observation_contains_info": boolean value of whether the information is available in the current "Observation";
}
"""
        prompt = gen_basic_prompt("", task_domain, function_description=function_description)
        prompt += observation_prompt.format(actree=actree)
        prompt += observation_description_prompt
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += info_needed_prompt
        prompt += whether_call_info_extractor_instruction
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt

        return prompt
    
    @staticmethod
    def subtasks_in_plan(task_content, task_domain, expectation, plan, finished_subtasks, not_stop_reason, info_needed,):
        logging.info("===== Currentprompt: gen_whether_call_info_extractor_prompt.subtasks_executed =====")

        # special prompts
        function_description = """You should decide whether the subtasks in the "Rough Plan" need to be executed. \n"""
        maintask_expectation_prompt = f"""### Expectation
{expectation}
"""
        rough_plan_prompt = f"""### Rough Plan
{plan}Note: The plan may contain speculative or 'hallucinated' subtasks, due to limited information available at the time of planning.

"""
        if finished_subtasks.strip() == "":
            finished_subtasks = """Currently, there are no finished subtasks, meaning that you have just started the task."""
        finished_subtasks_prompt = f"""### Finished Subtasks
Here are the finished subtasks conducted by yourself, note that it is not guaranteed that all the subtasks are completed, the completion status are also given:
{finished_subtasks}
"""
        not_stop_reason_prompt = f"""### Reason for not stopping
Focus on the last subtask, since it leads you to the current situation. You have analyzed the reason for not stopping the task:
{not_stop_reason}

"""
        info_needed_prompt = f"""### Information Needed
{info_needed}

"""
#         contain_info_analysis = f"""### Whether the Information is Contained in the Observation
# {contain_info_str}

# """
        whether_call_info_extractor_instruction = f"""### Reasoning Process
You are going to extract the information needed by the "Main Task" and "Information Needed". For each subtask in the "Rough Plan", consider followings:
    1. If the subtask won't be executed, is the information still in a correct manner? e.g. if you haven't sort the list, you may extract the wrong information.
    2. Does it also aim to find the info: {info_needed}?  If so, it might duplicate the current info-extraction subtask, making it redundant and unnecessary.
"""
        if not_stop_reason.strip() != "":
            whether_call_info_extractor_instruction += f"""    3. Does the "Reason for not stopping" provide you with any insights on whether you can extract the information?
"""
        whether_call_info_extractor_instruction += """Analyze the necessity of each subtask in the "Rough Plan" and output your reasoning after the key "reasoning_of_plan". 
Then output True/False after the key "necessary_subtask_in_plan" to indicate whether there are still necessary subtasks in "Rough Plan" that need to be executed.

"""
        format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "reasoning_of_plan": string of your analysis, whether EACH subtask in the "Rough Plan" is necessary to be executed before extracting the proper information, or whether it is redundant;
    "necessary_subtask_in_plan": boolean value, True if there are still necessary subtasks in "Rough Plan" that need to be executed, False otherwise;
}
"""
        prompt = gen_basic_prompt("", task_domain, function_description=function_description)
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += finished_subtasks_prompt
        prompt += rough_plan_prompt
        if not_stop_reason.strip() != "":
            prompt += not_stop_reason_prompt
        prompt += info_needed_prompt
        prompt += whether_call_info_extractor_instruction
        prompt += gen_prior_knowledge_emphasize_prompt(prompt)
        prompt += format_instructions_prompt

        return prompt
    
    @staticmethod
    def reasoning(task_content, task_domain, expectation, actree, observation_description, 
                                            plan, finished_subtasks, ScratchPad_Info, not_stop_reason):
        """
        ask the agent whether to call the info extractor
        """
        logging.info("===== Currentprompt: gen_whether_call_info_extractor_prompt =====")

        # special prompts
        maintask_expectation_prompt = f"""### Expectation
Here is your expectation for finishing the task:
{expectation}

"""
        observation_description_prompt = f"""### Observation Description
The description of the current page is:
{observation_description}

"""
        rough_plan_prompt = f"""### Rough Plan
{plan}
Note: The plan may contain speculative or 'hallucinated' subtasks, due to limited information available at the time of planning.

"""
        if finished_subtasks.strip() == "":
            finished_subtasks = """Currently, there are no finished subtasks, meaning that you have just started the task."""
        finished_subtasks_prompt = f"""### Finished Subtasks
Here are the finished subtasks conducted by yourself, note that it is not guaranteed that all the subtasks are completed, the completion status are also given:
{finished_subtasks}

"""
        empty_ScratchPad_Info = """### ScratchPad Information
Currently, there is no information in the ScratchPad. If the "Criteria" needs some answer to be stored, it is possiblly you have not found it, which means you cannot stop the task now.
"""
        not_stop_reason_prompt = f"""### Reason for not stopping
Focus on the last subtask, since it leads you to the current situation. You have analyzed the reason for not stopping the task:
{not_stop_reason}

Take it into consideration and decide whether to redo the last subtask or refine the plan.

"""    
        whether_call_info_extractor_instruction = f"""### Reasoning Process
**Information Requirements**:
What kind of information do you need? Consider the following aspects:
    1. What does the "Main Task" and "Expectation" require? Is it a single piece of information, or multiple pieces? 
    2. If it is multiple pieces, think about the "ScratchPad Information", which contains the information you have collected. What else is needed to fulfill the "Main Task" and its "Expectation"?
Describe the information, including following aspects:
    1. Information Description: a simple and clear description of this information, output after the key "info_description";
    2. Information Requirements: 
        - Should the info be concise or detailed? For example, a direct command is needed for 'Give me the command to clone a repository,' whereas a question like 'Compare the time of driving and walking' demands a more elaborate answer detailing both the mode of transport and the respective durations.
        - What kind of info is needed? For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name.
        Output the requirements after the key "info_requirements".

**Whether Info Extraction**:
Judge whether you can extract the information you need based on the current situation. Consider the following:
    1. Does the current "Observation" directly contain the information, which is needed by the "Main Task" and its "Answer requirements"? Does the "Observatoin Description" shows that the current page is the right page to extract the information? Note that you can't interact with the elements on the page, i.e. you can only observe it and scroll to view it. So don't make judgments based on the links on the page. Output your reasoning after the key "reasoning_of_observation" why this page includes the information needed. If you think the information is directly available on the current page, output True after the key "is_info_extraction", otherwise False.
    2. Does the "Rough Plan" contain some necessary subtasks that should be executed before you can extract the information? i.e. if it/they are not executed, you are not in the right page to extract the information. For example, if you haven't sort the list, you may extract the wrong information (in another ordoer). Output your reasoning why each subtask in the "Rough Plan" is necessary or not after the key "reasoning_of_plan". If there is still necessary subtasks, you should not start extracting information.
    3. Think about the "ScratchPad Information", which contains the information you have collected. What else is needed to fulfill the "Main Task" and its "Expectation", and can it be found in the current page? Output your reasoning after the key "reasoning_of_scratchpad".
    4. Does the "Reason for not stopping" provide you with any insights on whether you can extract the information? Output your reasoning after the key "reasoning_of_not_stop".
Summary your reasoning process after the key "reasoning_process". If you decide to extract information now, output True after the key "is_info_extraction", otherwise False. If you decide not, output the reason after the key "not_info_extraction_reason".

"""
        format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "info_description": string of simple, clear description of the information you want to extract from the current page;
    "info_requirements": string of the concrete requirements of the information, concise or detailed, what kind of info it is;
    "reasoning_of_observation": string of your reasoning, whether the desired information is directly available in the current "Observation";
    "observation_contains_info": boolean value of whether the information is available in the current "Observation";
    "reasoning_of_plan": string of your reasoning, whether the subtasks in the "Rough Plan" are necessary to be executed before extracting the proper information;
    "necessary_subtask_in_plan": boolean value, True if there are still necessary subtasks in "Rough Plan" that need to be executed, False if all necessary subtasks are finished;
    "reasoning_of_scratchpad": string of your reasoning, what information is needed besides the "ScratchPad Information" to fulfill the "Main Task" and its "Expectation";
    "reasoning_process": string of your reasoning, whether to start an info extraction subtask;
    "is_info_extraction": boolean value of whether to start an info extraction subtask;
    "not_info_extraction_reason": string of the reason why you decide not to start an info extraction subtask;
}
"""
        prompt = gen_basic_prompt("", task_domain)
        prompt += observation_prompt.format(actree=actree)
        prompt += observation_description_prompt
        prompt += maintask_prompt.format(task_content=task_content)
        prompt += maintask_expectation_prompt
        prompt += finished_subtasks_prompt
        prompt += rough_plan_prompt
        if ScratchPad_Info.strip() != "":
            prompt += scratchpad_info_prompt.format(ScratchPad_Info=ScratchPad_Info)
        else:
            prompt += empty_ScratchPad_Info
        prompt += not_stop_reason_prompt
        prompt += whether_call_info_extractor_instruction
        prompt += format_instructions_prompt

        return prompt
    
    @staticmethod
    def judging(reasonings: dict):
        """
        after generating the reasoning, use it to generate the final judging prompt of the boolen flags
        """
        reasonings_str = f"""**reasoning_of_observation**: {reasonings["reasoning_of_observation"]}
**reasoning_of_plan**: {reasonings["reasoning_of_plan"]}
"""
        output_flags = """
    "observation_contains_info": boolean value, True if the "reasoning_of_observation" indicates that the desired information is available in the current "Observation";
    "necessary_subtask_in_plan": boolean value, True if the "reasoning_of_plan" indicates that there are still necessary subtasks in the "Rough Plan" that need to be executed, False if all necessary subtasks are finished;
"""
        prompt = gen_basic_judging_prompt(reasonings_str, output_flags)

        return prompt 
    
# ==================== gen_answer_description_prompt ====================
def gen_answer_description_prompt(answer_requirements, domain):
    """
    transform the requirements into a phrase
    """
    logging.info("===== Currentprompt: gen_answer_description_prompt =====")

    # special prompts
    function_description = """You should transform the "Answer Requirements" into a phrase precisely and without losing any information. \n"""
    answer_requirements_prompt = f"""### Answer Requirements
The requirements of the answer are:
{answer_requirements}
"""
    answer_description_instruction = """### Reasoning Process
**Info Phrase Extraction**:
    Transform the "Answer Requirements" into a phrase after the key "info_needed".
**Info Requirements**:
    - Should the info be concise or detailed? For example, a direct command is needed for 'Give me the command to clone a repository,' whereas a question like 'Compare the time of driving and walking' demands a more elaborate answer detailing both the mode of transport and the respective durations.
    - What kind of info is needed? For example, "Get the driving time from CMU to airport" needs a specific time. "Tell me the name of the user who made the most commits to the repository" needs a name.
    Output the requirements after the key "info_requirements".

"""
    format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "info_needed": string of the phrase that meets the "Answer Requirements";
    "info_requirements": string of the concrete requirements of the information, concise or detailed, what kind of info it is;
}
"""
    prompt = gen_basic_prompt("", domain, function_description=function_description)
    prompt += answer_requirements_prompt
    prompt += answer_description_instruction
    prompt += format_instructions_prompt
    return prompt