import logging
from .prompts import *


# ========== basic prompt =========
basic_prompt_without_info = """You are an AI agent tasked with reviewing the current webpage and extracting the required information. \n"""
AGENT_ROLE = "extractor"

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

# ==================== gen_info_extraction_prompt ====================
def gen_info_extraction_prompt(task_content, domain, expectation, actree, observation_description,
                               finished_subtasks_str,
                               executed_scroll_actions, reflection):
    """
    used for generating the final answer of a subtask
    """
    logging.info("===== Currentprompt: gen_info_extraction_prompt get_info =====")

    # special prompts
    function_description = """You need to extract the final answer from the current observation. \n"""
    observation_without_info_prompt = f"""### Observation
The current observation of the webpage is:
{actree}

"""
    expectation_prompt = f"""### Criteria
Here is the demand for a final answer:
{expectation}
"""
    finished_subtasks_prompt = f"""### Finished Subtasks
To get the information required by the "Main Task", you have already done the following opeartions:
{finished_subtasks_str}So, the only thing left is to extract the final answer from the current observation.

"""
    executed_scroll_actions_prompt = f"""### Executed Scroll Actions
Here are the executed scroll actions:
{executed_scroll_actions}

"""
    reflection_prompt = f"""### Reflection
Here is the reflection of the current state, which may provide some insights for your decision:
{reflection}

"""
    final_answer_prompt = """### Reasoning Process
**Generate the Final Answer**:
Generate the final answer. Consider the following aspects:
    - Where does the answer lie in the "Observation"? How does it meet the "Criteria"? Output your reasoning after the key "reasoning_of_answer".
    - According to "Criteria", if you can find / you already found the answer in the current "Observation", output True after the key "stop_flag", otherwise False.
    - If you can, extract it and output your final answer after the key "answer".
    - If you can't, first consider scrolling the page under the following instructions. However if you have scrolled multiple times and the answer is still not found, the answer might not exist. Output "N/A" or '0' after the key "answer" according to the "Answer Requirements" and output True after the key "stop_flag".

**Scroll the Page**:
    - If you cannot find the answer in the current "Observation", maybe the desired information lies in other parts of the webpage. Do you need to scroll up or down to find the answer? Consider the "Executed Scroll Actions", where you have already scrolled the page. Do you need to further scroll down or scroll up back? Output your reasoning after the key "scroll_reason".
    - If you indeed need to scroll, output True after the key "scroll_flag", otherwise False. 
    - Finally, output your direction of scrolling, "up" or "down", after the key "scroll_direction".

"""
    format_instructions_prompt = """Format your response in JSON, including the following keys:
{
    "reasoning_of_answer": string of your reasoning, where is the answer located in the "Observation";
    "stop_flag": boolean value of whether you need to stop the search, either you found the answer or you are sure that the answer doesn't exist
    "answer": string of the final answer, satisfying the "Criteria"; If the answer doesn't exist, output "N/A" or similar non-existing answer;
    "scroll_reason": string of your reason if you need to scroll the page to find the answer;
    "scroll_flag": boolean value of whether you need to scroll the page to find the answer;
    "scroll_direction": "up" or "down" of the direction you need to scroll;
}
"""
    prompt = basic_prompt_without_info
    prompt += function_description
    prompt += observation_without_info_prompt
    prompt += maintask_prompt.format(task_content=task_content)
    prompt += expectation_prompt
    if finished_subtasks_str.strip() != "":
        prompt += finished_subtasks_prompt
    if executed_scroll_actions.strip() != "":
        prompt += executed_scroll_actions_prompt
    if reflection.strip() != "":
        prompt += reflection_prompt
    prompt += final_answer_prompt
    prompt += gen_prior_knowledge_emphasize_prompt(prompt)
    prompt += format_instructions_prompt
    return prompt
