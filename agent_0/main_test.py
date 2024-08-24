
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import re
import json

sys.path.append(os.getcwd())

from utils import llm_request, LOGGER
import agent  # 从 agent 模块中导入 MedicalAgent
from sub_agent import decomposition

def solve_problem(user_problem):
    agent_tools =[
        {
            "type": "function",
            "function": {
                "name": "affinity_agent",
                "description": "To predict the drug's affinity against the protein, ask the Affinity Agent for information regarding the drug's binding affinity to the target protein. Given drug SMILES and protein amino acid sequence, return the predicted binding affinity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_smiles": {
                            "type": "string",
                            "description": "The Drug SMILES",
                        },
                        "protein_sequence": {
                            "type": "string",
                            "description": "The Protein Amino Acid Sequence",
                        }
                    },
                    "required": ["drug_smiles", "protein_sequence"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "admet_agent",
                "description":'''
                Given the drug SMILES, use a pretrained model to predict the ADMET properties of the drug. 
                This includes predictions related to absorption, distribution, metabolism, excretion, and toxicity. 
                The agent will be triggered by any inquiries regarding ADMET-related aspects of the drug.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_smiles": {
                            "type": "string",
                            "description": "The Drug SMILES",
                        }
                    },
                    "required": ["drug_smiles"],
                },
            }
        },
    ]
    LOGGER.log_with_depth(f"User problem:\n{user_problem}", depth=0)
    
    LOGGER.log_with_depth(f"Planing...", depth=0)
    time.sleep(2)
    LOGGER.log_with_depth(f"[Thought] Least to Most Reasoning: Decompose the original user problem", depth=0)
    decomposed_resp = decomposition(user_problem, agent_tools)

    subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
    subproblems = [subproblem.strip() for subproblem in subproblems]
    
    for idx, subproblem in enumerate(subproblems):
        LOGGER.log_with_depth(f"<subproblem>{subproblem}</subproblem>", depth=0)

    problem_results = []

    MedicalAgent = agent.MedicalAgent(user_problem, depth=1)

    LOGGER.log_with_depth(f"[Action] Solve each subproblem...", depth=0)
    for sub_problem in subproblems:
        LOGGER.log_with_depth(f"Solving...", depth=0)
        response = MedicalAgent.request(f"The original user problem is: {user_problem}\nNow, please you solve this problem: {sub_problem}")

        # LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=0)
        problem_results.append(response)
    problem_results = '\n'.join(problem_results)
    
    messages = []

    LOGGER.log_with_depth("Thinking...")
    time.sleep(2)
    LOGGER.log_with_depth("[Action] Requesting the final result from the Reasoning Agent...", depth=0)
    

    system_prompt = f'''
You are an expert in medical. Based on your own knowledge and the sub-problems solved, please solve the user's problem and provide the reason.
First, analyze the user's problem.
Second, present the final result of the user's problem in <final_result></final_result>. For a binary problem, it is a value between 0 to 1. You must include the exact probability within the <final_result></final_result> tags, e.g., 'The failure rate of the clinical trial is <final_result>0.8</final_result>.'
Third, explain the reason step by step.
Note: You must include the exact probability within the <final_result></final_result> tags.
    '''

    messages.append({ "role": "system", "content": system_prompt})
    messages.append({ "role": "user", "content": f"The original user problem is: {user_problem}"})
    messages.append({ "role": "user", "content": f"The subproblems have solved are: {problem_results}"})
    messages.append({ "role": "user", "content": "Please solve the user's problem and provide the reason."})

    final_results = llm_request(messages)
    
    LOGGER.log_with_depth(final_results.choices[0].message.content, depth=0)
    LOGGER.log_with_depth("\n======================    END   ================\n", depth=0)

    return problem_results, final_results.choices[0].message.content




if __name__ == "__main__":
    user_prompt = "Predict the DTI of aspirin and ibuprofen on proteins p53 and BRCA1."
    subproblem_solutions, final_solution = solve_problem(user_prompt)
    print("Subproblem Solutions:")
    print(subproblem_solutions)
    print("Final Solution:")
    print(final_solution)