from typing import Any
from graphviz import Digraph
import textwrap

# if True, visualization will be updated each time a new node is added. False for just update every rollout
FREQUENT_VIEW = True

# define the color for the different status of node
# YELLOW-current_path, RED-current_focus, GREEN-simulation, BLUE-terminal
# color name pages: https://graphviz.org/doc/info/colors.html

"""
===== color legend =====
yellow: current_path
red: current_focus
green: simulation
deepskyblue: valid_terminal
gold2: invalid_terminal
gray66: simulation_node
chocolate1: to_be_deleted
green1: final_terminal
========================
"""

# root, node should be Node_cls
# a class work for visualization
class MCTS_vis:
    def __init__(self, root, WebTask_content, SubTask_content, SubTask_expectation, filename="mcts"):
        self.G = Digraph(filename, filename)
        root_pic_path = f"node_pics/{root.name}.pdf"
        self.G.node(root.name, 'ROOT\nMain: '+WebTask_content+'\nSub: '+SubTask_content+'\nExpectation: '+SubTask_expectation,
                    URL=root_pic_path)
        # self.G.node('legend', 'LEGEND\nYELLOW-current_path\nRED-current_focus\nGREEN-simulation', shape="box")

    def view(self):
        self.G.view()

    def add_node(self, node):
        label = f"Node-{node.name[:4]}\nvisit_counts={node.visit_counts}\nM={node.M}\nQ={node.Q}"
        pic_path = f"node_pics/{node.name}.pdf"
        self.G.node(node.name, label,URL=pic_path)
        self.edge_update(node)
        self.view() if FREQUENT_VIEW else None

    def node_update(self, node):
        label = f"Node-{node.name[:4]}\nvisit_counts={node.visit_counts}\nM={node.M}\nQ={node.Q}\nScore={node.reward}\nself_reward={node.reward_self}"
        label += f"\nSeperate_Score = {node.reward_separate[0]} + {node.reward_separate[1]}"
        if hasattr(node, "terminal_reward"):
            label += f"\nTerminal_reward={node.terminal_reward}"
        self.G.node(node.name, label)
        self.view() if FREQUENT_VIEW else None

    def edge_update(self, node, line_style="solid"):
        """
        set the readable action name according to action_types
        """
        try:
            action_type = node.src_action["action_type"]
            if action_type == "type":
                readable_action = action_type + " " + repr(node.src_action["action_str"]) + "\n" + repr(node.src_action["element_info"])
            else:
                readable_action = action_type + " \n" + repr(node.src_action["element_info"])
        except:
            readable_action = "Not defined yet"

        readable_action = textwrap.fill(readable_action, width=50)

        self.G.edge(node.parent.name, node.name, label=readable_action, style=line_style)
        self.view() if FREQUENT_VIEW else None

    def node_path(self, node):
        self.G.node(node.name, style="filled", color="yellow")
        self.view() if FREQUENT_VIEW else None
    
    def node_focus(self, node):
        self.G.node(node.name, style="filled", color="red")
        self.view() if FREQUENT_VIEW else None

    def node_reset(self, node):
        # reset the node style, not the label
        self.G.node(node.name, shape="", style="", color="")
        self.view() if FREQUENT_VIEW else None

    def node_terminal(self, node, valid_terminal=False, force_stop=False):
        if valid_terminal:
            self.G.node(node.name, style="filled", color="deepskyblue")
        else:
            self.G.node(node.name, style="filled", color="gold2")
        if force_stop:
            label = f"FORCE_STOP\nNode-{node.name[:4]}\nvisit_counts={node.visit_counts}\nM={node.M}\nQ={node.Q}\nScore={node.reward}\nself_reward={node.reward_self}"
            label += f"\nSeperate_Score = {node.reward_separate[0]} + {node.reward_separate[1]}"
            self.G.node(node.name, label)
        
        self.view() if FREQUENT_VIEW else None

    def node_final_terminal(self, node):
        self.G.node(node.name, style="filled", color="green1")
        self.view() if FREQUENT_VIEW else None

    def node_delete(self, node):
        label = f"Node-{node.name[:4]}\nTo_be_deleted"
        pic_path = f"node_pics/{node.name}.pdf" # only works after transfered, using relative path
        self.G.node(node.name, label,URL=pic_path)
        self.G.node(node.name, style="filled", color="chocolate1")
        # construct edge
        self.G.edge(node.parent.name, node.name, lable=node.src_action, style="dashed")        
        self.view() if FREQUENT_VIEW else None


    def node_simulation_in(self, node, src_action, ):
        return None    
    def node_simulation_out(self,):        
        return None

    def node_simulation(self, node, src_action):
        """
        add a simulation node. Here the input node is the simulation node's parent node
        """
        label = f"Sim_Node\nScore={node.reward_sim}"
        label += f"\nSeperate_Score = {node.reward_sim_separate[0]} + {node.reward_sim_separate[1]}"
        sim_node_name = f"{node.name}_sim"
        pic_path = f"node_pics/{sim_node_name}.pdf" # only valid after transfered, using relative path
        self.G.node(sim_node_name, label, URL=pic_path, style="filled", color="gray66") # gray66 is gray, indicating a simulation node
        # self.edge_update(node)
        self.sim_edge_update(node.name, sim_node_name, src_action)
        self.view() if FREQUENT_VIEW else None   

        return None     

    def sim_edge_update(self, node_name, sim_node_name, src_action):
        """
        set the readable action name according to action_types
        """
        try:
            action_type = src_action["action_type"]
            if action_type == "type":
                readable_action = action_type + " " + repr(src_action["action_str"]) + repr(src_action["element_info"])
            else:
                readable_action = action_type + " " + repr(src_action["element_info"])
        except:
            readable_action = "Not defined yet"

        readable_action = textwrap.fill(readable_action, width=50)
        self.G.edge(node_name, sim_node_name, label=readable_action, style="dashed")
        self.view() if FREQUENT_VIEW else None

class Info_extractor_vis(MCTS_vis):     
    def __init__(self, root, WebTask_content, SubTask_content, SubTask_expectation, filename="mcts"):
        self.G = Digraph(filename, filename)
        root_pic_path = f"node_pics/{root.name}.pdf"
        self.G.node(root.name, 'ROOT\nMain: '+WebTask_content+'\nSub: '+SubTask_content+'\nExpectation: '+SubTask_expectation,
                    URL=root_pic_path)
        # add a sinlge node to show the task's information
        self.G.node("task_info", 'Task\nMain: '+WebTask_content+'\nSub: '+SubTask_content+'\nExpectation: '+SubTask_expectation, shape="box")

    def add_node(self, node,):
        label = f"Node-{node.name[:4]}"
        pic_path = f"node_pics/{node.name}.pdf" # only works after transfered, using relative path
        self.G.node(node.name, label,URL=pic_path)
        self.edge_update(node)
        self.view() if FREQUENT_VIEW else None

    def node_terminal(self, node, answer):
        # add a single node to show the answer
        self.G.node("answer", 'Answer: '+answer, shape="box")
        self.G.edge("task_info", "answer",)

        self.view() if FREQUENT_VIEW else None

