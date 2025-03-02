import json

from WebPilot.model import ask_LLM

# import other functions defined by ourselves
from WebPilot.prompter.planner import gen_answer_requirements_prompt

# ==================== SubTask_cls ====================
# define a subtask class to save the subtask information


class SubTask_cls:
    def __init__(self, content: str, idx: int) -> None:
        self.content = content
        self.idx = idx
        self.steps = 0
        self.trajectory = []
        self.executed_actions = ["",]
        self.stop_flag = False
        # self.final_answer = ""
        self.expectation = ""
        self.complete_flag = True
        self.only_info_extraction_flag = False
        self.interaction_type = "web_interaction"  # default is web_interaction

    def gen_answer_requirements(self, domain, actree):
        """
        Let the agent decide what kind of the final answer is needed. Add the response to the Expectation to let the agent know its target during search.
        Will be called after a SubTask is initialized.
        """

        need_answer = self.need_answer

        if repr(need_answer).lower() == "true":
            self.need_answer = True
            # generate the 'answer_requirements'
            prompt = gen_answer_requirements_prompt(
                self.content, domain, self.expectation,)
            response = ask_LLM(prompt)
            self.expectation += "\n**Answer Requirements**: " + \
                json.loads(response)["answer_requirements"]

        else:
            self.need_answer = False

        return None
