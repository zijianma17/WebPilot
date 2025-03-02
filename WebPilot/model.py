'''
here define and set the parameters for the LLM model
as well as some other parameters for the algorithm, maybe for the further ablation study
'''

import logging
import openai
import json

# set the default model
DEFAULT_MODEL = "3.5" # 3.5, 4o, 4omini

W_EXP = 1
REGULARIZE_FLAG = True
DECOMPOSE_FLAG = True
MCTS_VARIANT_FLAG = True
BACK_BY_RESET_FLAG = True
MAJORITY_VOTE_FOR_STOP_FLAG = True
MAX_SUBTASK_NUM = 15 
MAX_TERMINAL_NODES = 1
MAX_ITERATION_TIMES = 10

MAX_RECTIFY_TIMES = 2
WITH_EXPECTATION_FLAG = False
SKIP_COMPLETE_ESTIMATION_FLAG = False
BRANCH_NUM = 3

MODEL_TEMP = 0.3

# ==================== LLM model ====================
def ask_LLM(prompt, model=DEFAULT_MODEL, is_json_mode=True, model_temp=MODEL_TEMP):
     logging.info("==================== LLM is called ====================")

     # setting model and pricing, 3.5 as default
     model = str(model)
     if model == "3.5":
          LLM_model = "gpt-3.5-turbo-0125"
     elif model == "4o":
          LLM_model = "gpt-4o-2024-05-13"
     elif model == "4omini":
          LLM_model = "gpt-4o-mini-2024-07-18"

     model = LLM_model
     messages=[
          {"role":"user", "content":prompt},
     ]
     max_tokens = 4096

     logging.info("+++prompt content+++")
     logging.info(prompt)

     # add try-except to avoid the API call error, use while loop to retry, also add a max_retry_times
     try_times = 0
     max_retry_times = 3
     while True and try_times < max_retry_times:
          try:
               if is_json_mode:
                    response = openai.ChatCompletion.create(model=model,
                                                            temperature=model_temp,

                                                            max_tokens=max_tokens,
                                                            messages=messages,
                                                            response_format={"type":"json_object"})
                    
                    # test whether the output could be parsed with json
                    response_str = response.choices[0].message["content"]
                    _ = json.loads(response_str)

               else:
                    response = openai.ChatCompletion.create(model=model,
                                                            temperature=model_temp,
                                                            max_tokens=max_tokens,
                                                            messages=messages)
          except Exception as e:
               logging.error("Error in LLM call")
               logging.error(e)
               try_times += 1
               if try_times == max_retry_times:
                    logging.error(f"Retry {try_times} times")
               continue
          
          break
     
     logging.info("==================== Response content ====================")
     logging.info("============================================================\n")
     logging.info(response.choices[0].message["content"])
     logging.info("==================== LLM call finished ====================\n\n\n\n\n\n\n\n\n\n")

     return response.choices[0].message["content"]
