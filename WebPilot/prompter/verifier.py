import logging
from .prompts import *
from .executor import gen_action_space_prompt


# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with verifying the feasibility of an action generated by another AI agent. \n"""
AGENT_ROLE = "verifier"

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


# ==================== format_regularizer ====================
def gen_format_regularizer_prompt(actree, action_dict_str:str, warnings:str):
    """
    Used in AI-function, called in a loop until the action is in the right format.
    """
    logging.info("===== Currentprompt: gen_format_regularizer_prompt call for format =====")
    
    # special prompts
    function_description = """Now the action is in an incorrect format, which is not acceptable by the environment. Consider the "Warning" carefully and correct the action accordingly. \n"""

    current_action = f"""### Current Action Representation
The action you've generated is:
{action_dict_str} 

"""
    warning_prompt = f"""### Warnings
Here are some warnings regarding to your former generated action: 
{warnings}
Consider the "Warning" carefully and correct the action accordingly.

"""
    format_regularizer_instruction = """### Reasoning Process
The "Current Action Representation" is the action you've generated, but in an incorrect format, which is not acceptable by the environment.
1. Consider the "Warning" carefully and correct the action accordingly.
2. If the element_id is missing or incorrect, find it in the "Observation" according to the "element_info".
3. Refer to "Action Space" to output the correct action format.

"""
    format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "reasoning": string of your reasoning, why the action is incorrect and how you correct it;
    "action_type": string of the action type, choosing from the predefined action types;
    "action_str": string of the action-specific annotations, formatted according to the action space description. The element_id should be an integer.
    "element_info": the information after the element_id in the same line, which is the text of the element.
}

"""
    attention_prompt = """Attention: Use element_id, the output should be an integer."""

    prompt = basic_prompt_without_info
    prompt += function_description
    prompt += observation_prompt.format(actree=actree)
    prompt += current_action
    prompt += gen_action_space_prompt()
    prompt += format_regularizer_instruction
    if warnings.strip() != "":
        prompt += warning_prompt
    prompt += format_instructions_prompt
    prompt += attention_prompt

    return prompt

# ==================== execution_reflection for interact_verifier ====================
def gen_execution_reflection_prompt(action, former_actree, actree):
    """
    used in _env_step, ask the llm to judge whether the action is indeed executed.
    point out the changes of the actree,
    reflect on the given action
    """
    logging.info("===== Currentprompt: gen_execution_reflection_prompt, interact_verifier to execution_reflection =====")

    # special prompts
    instruction_prompt = """You are an AI agent in a web page. You should accomplish some task by interacting with the web page.
After one action, the env should change according to your action. 
"""
    former_observation_prompt = f"""### Former Observation
The observation of the former state is: 
{former_actree}

"""
    action_verficator_prompt = f"""### Former Action
The former action (decided by you) is: 
{action}

This action is failed to execute. Possible reasons could be due to the incorrect element_id (where the elment is not interactable), or the wrong pair of action_type and action_str.
Focus on the "Former Observation", and the "Former Action" you've generated, analyze why the action is not executed, is it due to the non-interactable element or the wrong action_type and action_str? Give me your reflection.

"""
    format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "execution_reflection": if not succeed, reflection on the given action.
}

"""

    prompt = instruction_prompt
    prompt += former_observation_prompt
    prompt += observation_prompt.format(actree=actree)
    prompt += gen_action_space_prompt()
    prompt += action_verficator_prompt
    prompt += format_instructions_prompt

    return prompt

# ==================== gen_general_execution_reflection_prompt ====================
def gen_general_execution_reflection_prompt(new_action, failed_action, failed_execution_reflection):
    """
    generate a action-relevant reflection to let the agent avoid similar mistakes.
    """
    logging.info("===== Currentprompt: gen_general_execution_reflection_prompt =====")

    prompt = f"""You are an AI agent in a web page. You should accomplish some task by interacting with the web page.
You should generate some action according to the observation. After the action, the env should change according to your action. Following are the action space introduction.
"""
    prompt += gen_action_space_prompt()
    prompt += f"""### Failed Action
You have generated an action, which is unfeasible:
{failed_action}

"""
    prompt += f"""### New Action
After your further trial, you have generated a new action, which succeeds:
{new_action}

"""
    prompt += f"""### Failed Execution Reflection
Here is the reflection on the failed action generated by you:
{failed_execution_reflection}

"""
    prompt += """Now please reflect on the failed action, and give me some advice to avoid similar mistakes in the future. Don't be so specific, consider the general situation.
"""
    format_instructions_prompt = """Format your response in JSON, including the following key:
{
    "general_execution_reflection": a short sentence that records your general reflection.
}

"""
    
    return prompt+format_instructions_prompt

# ==================== re_gen_action_prompt ====================
def gen_re_gen_action_prompt(former_gen_action_prompt, failed_execution_reflection, failed_action):
    """
    in action verifier, when the action is not executed, ask the agent to re-generate the action.
    """
    logging.info("===== Currentprompt: gen_re_gen_action_prompt =====")

    prompt = former_gen_action_prompt
    prompt += f"""### Failed Execution Reflection
Here is the reflection on the failed action generated by you:
{failed_execution_reflection}

"""
    prompt += f"""### Failed Action
The failed action is:
{failed_action}

The "action_intent" is correct, but the "action_str" or the matching of "action_type" and "action_str" is wrong, which leads to the failure of the action execution.
Don't generate the same action again.
Think about the attribute of the "element_info", is it interactable? If not, use some other element_id nearby, but remain the "action_intent".

"""
    prompt += """Now please re-generate the action. Output format refer to the above instruction.
"""
    
    return prompt

# ==================== element_info_alignment_warning_prompt ====================
def gen_element_info_alignment_warning_prompt(reasoning_process, action_intent, element_choice,
                                              target_line, element_info):
    
    warning = f"""**Element Alignment Warning**
Your former **reasoning_process** is:
{reasoning_process}
Your former **action_intent** is:
{action_intent}"""
    if element_choice.strip() != "":
        warning += f"""
Your analysis of which element to interact with is:
{element_choice}"""
    warning += f"""
You have already generate the action once. But the element_id and element_info are not aligned.
The two options are:
1. {target_line};
2. {element_info}.
Think about the "reasoning_process" and "action_intent" you have, which one is correct to fulfill your action_intent?
Please re-generate the action with the correct element_id and element_info from the "Observation".

"""
    
    return warning