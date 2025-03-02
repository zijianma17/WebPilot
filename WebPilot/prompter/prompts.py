
import logging
import os
import yaml

# =================== basic components of prompts for a single tree search ===================

# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with automating interactions on a webpage. \n"""
maintask_prompt = """### Main Task
{task_content}

"""

# ========== basic prompt with prior knowledge =========
def gen_basic_prompt_origin(task_content, domain, agent_role, function_description=""):
    """
    generate a basic prompt, w.r.t the domain
    Args:
        task_content : task description, could be main_task or subtask; if "", then use basic_prompt_without_info
        domain (_type_): gitlab, map (currently)

    """
    # construct the prior knowledge according to the domain
    file_path = os.path.join(os.path.dirname(__file__), "prior_knowledge.yaml")

    with open(file_path, "r") as f:
        prior_knowledge_all = yaml.load(f, Loader=yaml.FullLoader)

    prior_knowledge = prior_knowledge_all[domain][agent_role]

    # it's currently a list, convert it to a string with index and newline
    if isinstance(prior_knowledge, list):
        prior_knowledge_str = "".join([f"{idx+1}. {item}" for idx, item in enumerate(prior_knowledge)])
    else:
        prior_knowledge_str = ""

    prompt = basic_prompt_without_info
    if function_description.strip() != "":
        prompt += function_description
    if task_content.strip() != "":
        prompt += maintask_prompt.format(task_content=task_content)
    if prior_knowledge_str.strip() != "":
        prompt += f"""### Prior Knowledge
The following are the prior knowledge you should consider which could help you to complete the task:
{prior_knowledge_str}
"""
    return prompt


# ========== get prior_knowledge according to domain =========
def gen_prior_knowledge_prompt_origin(domain, agent_role):
    file_path = os.path.join(os.path.dirname(__file__), "prior_knowledge.yaml")
    with open(file_path, "r") as f:
        prior_knowledge_all = yaml.load(f, Loader=yaml.FullLoader)

    prior_knowledge = prior_knowledge_all[domain][agent_role]
    prior_knowledge_str = "".join([f"{idx+1}. {item}" for idx, item in enumerate(prior_knowledge)])

    prompt = f"""### Prior Knowledge
The following are the prior knowledge you should consider which could help you to complete the task:
{prior_knowledge_str}
"""
    if prior_knowledge_str == "":
        prompt = ""

    return prompt

def gen_prior_knowledge_emphasize_prompt(current_prompt):
    if "### Prior Knowledge" in current_prompt:
        emphasize_prompt = """Please always pay attention to the given "Prior Knowledge". """
    else:
        emphasize_prompt = ""
    
    return emphasize_prompt


# ========== ScratchPad ==========
scratchpad_info_prompt = """### ScratchPad Information
The Information saved in the ScratchPad is:
{ScratchPad_Info}
"""

# ========== Observation ==========
observation_prompt = """### Observation
{actree}

"""

# ========== executed actions ==========
executed_action_prompt = """### Executed Actions
{executed_actions}
"""

# ==================== single observation description ====================
def gen_single_observation_description_prompt(task_content, actree):
    """
    Generate the prompt for observation description.
    """
    logging.info("===== Currentprompt: gen_single_observation_description_prompt =====")
    
    observation_description_prompt = f"""### Reasoning Process
**Overall Observation**: 
Describe the "Observation" consider following aspects:
	- What does the page look like? What is the main content?
	- What information is most critical? 
Output the observation after the key "overall_description".

**Top/Side Bars**:
	- Which elements are the top bar of the webpage, which elements are the sidebar? Note sidebar is not always available in some webpages.
Output after the key "top_side_bars".

**Main Body**:
	- What is the main body of the webpage? The main body of the observation locates usually after the 'main' element. Try to list the elements in the main body more detailed. 
Output after the key "main_body".

**Possible Interactable Elements**:
Which elements in the environment might be relevant or useful to interact with to completing the "Main Task"? Please list them in detail, and explain the possible effects of interacting with them. 
Output after the key "interactable_elements".

**Task-specific Elements Status**:
Which elements could reflect the progress of the "Main Task"? e.g. specific dropdown menu should be expanded(list the presented options of the dropdown menu), specific tab should be focused, specific text should be filled with some content. Please list them in detail.
Output after the key "task_specific_elements_status".

"""
    format_instructions_prompt = """Format your response in Json, including the following:
{
    "overall_description": string that describes the current observation overall;
    "top_side_bars": string that lists the top bar and sidebar elements, note that sidebar is not always available;
    "main_body": string that describes the main body of the webpage;
    "interactable_elements": string that lists the possible interactable elements and their effects;
    "task_specific_elements_status": string that lists the task-specific elements and their status.
}
"""
    prompt = basic_prompt_without_info
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += observation_prompt.format(actree=actree)
    prompt += observation_description_prompt
    prompt += format_instructions_prompt

    return prompt


def gen_basic_judging_prompt(reasonings, output_flags):
    prompt = f"""You are required to make evaluation as true/false based on the reasoning I give you. For each of the reasoning, there is a corresponding output key.
Here are the reasonings:
{reasonings}
Based on these reasonings, you need to output final judgements. The final judgements include the followings and should be output in JSON format:
{{{output_flags}}}
"""
    return prompt