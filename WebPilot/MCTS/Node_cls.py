import logging
from PIL import Image # for screenshot saving
import uuid # for uuid of each node
import copy

# ==================== Node_cls ====================
# define a single node class in MCTS
class Node_cls():
     def __init__(self, depth:int, actree:str, url:str, parent, src_action:dict, max_depth:int=10) -> None:
          self.name = str(uuid.uuid4()) # generate a uuid for each node
          self.parent = parent
          self.children = []
          self.visit_counts = 0
          self.M = 0 
          self.Q = 0 
          self.depth = depth
          self.actree = actree
          self.url = url
          self.src_action = src_action # dict: {"action_type":, "action_str":}
          self.max_depth = max_depth # if the depth is deep enough, stop deeper expansion
          self.terminal = False # whether the node is terminal
          self.answer = ""

          return None
     
     def update_rapid_access(self):
          """
          update the rapid_access dict for the node, using function because of the "EMPTY ACTION" nodes can't be updated directly after initialization
          The rapid_access dict is used for the rapid access of the node, including the last_url and the path from the last_url to the node.
          This function must only be called after the node.src_action is updated.
          """
          logging.info(f"===== Update the rapid_access dict for the node: {self.name[:6]} =====")
          # init the rapid access dict
          self.rapid_access = {"last_url":"", "path":[],}

          if self.parent is None: # the very first root node has no parent
               self.rapid_access["last_url"] = self.url 
          elif (    (self.url != self.parent.url) and 
                    ("http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/" not in self.url) # in map domain we don't use the url based rapid access
               ):
               self.rapid_access["last_url"] = self.url
          else:
               self.rapid_access["last_url"] = self.parent.rapid_access["last_url"]
               self.rapid_access["path"] = copy.deepcopy(self.parent.rapid_access["path"])
               if not "root" in self.name:
                    self.rapid_access["path"].append(self.src_action)
          
          log_str = "===== Rapid access updated: =====\n"
          log_str += f"Last accessible url: {self.rapid_access['last_url']}\n"
          if self.rapid_access["path"] == []:
               log_str += "No action needed to reach the node.\n"
          for idx, action in enumerate(self.rapid_access["path"]):
               log_str += f"""Action {idx+1}: {action["action_type"]} | {action["action_str"]} | {action["element_info"]}\n"""
          log_str += "========================================"

          logging.info(log_str)

          return None

     
     def _gen_children_empty(self, WebTask, SubTask, sample_num:int):
          """
          Here we firt generate empty childrens for selction to make the reflection available for sibling node generation.
          Generate some empty nodes, and generate action only when entering it.
          """
          action_dict = "EMPTY ACTION"
          for _ in range(sample_num):
               new_node = Node_cls(depth=self.depth+1, actree="", url="", parent=self, src_action=action_dict)
               self.children.append(new_node)
          return None

     def get_sibling_actions(self) -> list:
          """
          get the actions of its siblings, will be used for EMPTY nodes, which are generated when their parent is first expanded.
          return a string of executed actions, where each action is a dict including "action_type" and "element_info", "action_intent".
          """
          if not "root" in self.name: # root node has no siblings
               actions = []
               for sibling in self.parent.children:
                    if sibling is not self and sibling.src_action != "EMPTY ACTION":
                         actions.append(sibling.src_action)
               return actions
          else:
               return None

     def get_sibling_reflections(self) -> str:
          """
          Get the reflection of its siblings, will be used for EMPTY nodes, which are generated when their parent is first expanded.
          These reflections are generated w.r.t corresponding action_intent & observation_description & score_reason.
          """
          reflections = ""
          if not "root" in self.name: # root node has no siblings
               idx = 0
               for sibling in self.parent.children:
                    if sibling is not self and hasattr(sibling, "node_reflection"):
                         if sibling.node_reflection_for_sib.strip() != "":
                              idx += 1
                              reflections += str(idx)+". "+sibling.node_reflection_for_sib + "\n"
          return reflections
     
     def get_sibling_actions_n_reflections(self):
          """
          Get the actions with corresponding reflections of its siblings, will be used for EMPTY nodes, which are generated when their parent is first expanded.
          """
          sibling_actions_n_reflections = []
          if not "root" in self.name: # root node has no siblings
               for sibling in self.parent.children:
                    if sibling is not self and sibling.src_action != "EMPTY ACTION":
                         sibling_actions_n_reflections.append(sibling.node_reflection_for_sib)
                    
          return sibling_actions_n_reflections

     def get_former_node_reflections(self) -> str:
          """
          get the reflections of the former nodes, i.e. the nodes from the root to the parent node
          """
          reflections = []
          node = self.parent
          idx = 0
          while not "root" in node.name:
               idx += 1
               reflections.insert(0, str(idx)+". "+node.node_reflection_for_child)
               node = node.parent
          reflections = "".join(reflections)
          return reflections

     def _is_terminal(self):
          # judge whether the node is terminal
          if self.src_action["action_type"] == "stop":
               return True
          elif self.depth >= self.max_depth:
               return True
          else:
               return False

     def save_img(self, img):
          # save the current screenshot named by the node name
          img = Image.fromarray(img)
          img.save(f"webpilot_utils/node_pics/{self.name}.pdf")

     def save_sim_img(self, img):
          img = Image.fromarray(img)
          img.save(f"webpilot_utils/node_pics/{self.name}_sim.pdf")
