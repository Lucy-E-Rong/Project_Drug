
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import re
import json

sys.path.append(os.getcwd())

from utils import llm_request, LOGGER
from agent import MedicalAgent  # 从 agent 模块中导入 MedicalAgent

def solve_problem(user_problem):
    LOGGER.log_with_depth(f"User problem:\n{user_problem}", depth=0)

    # Initialize MedicalAgent
    medical_agent = MedicalAgent(user_problem, depth=1)
    # Using MedicalAgent to handle the problem
    drug_smiles = "Your_SMILES_here"  # 您需要根据实际情况填写
    protein_sequence = "Your_Protein_Sequence_here"  # 您需要根据实际情况填写
    result = medical_agent.affinity_agent(drug_smiles, protein_sequence)

    subproblem_solutions = result
    
    LOGGER.log_with_depth("[Action] Requesting the final result from the Reasoning Agent...", depth=0)

    system_prompt = f'''
You are an expert in medical. Based on your own knowledge and the sub-problems solved, please solve the user's problem and provide the reason.
First, analyze the user's problem.
Second, present the final result of the user's problem in <final_result></final_result>. For a binary problem, it is a value between 0 to 1. You must include the exact probability within the <final_result></final_result> tags, e.g., 'The failure rate of the clinical trial is <final_result>0.8</final_result>.'
Third, explain the reason step by step.
Note: You must include the exact probability within the <final_result></final_result> tags.
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The original user problem is: {user_problem}"},
        {"role": "user", "content": f"The subproblems solved are: {subproblem_solutions}"},
        {"role": "user", "content": "Please solve the user's problem and provide the reason."}
    ]

    final_results = llm_request(messages)
    
    LOGGER.log_with_depth(final_results.choices[0].message.content, depth=0)
    LOGGER.log_with_depth("\n======================    END   ================\n", depth=0)

    return subproblem_solutions, final_results.choices[0].message.content

def solve_problem_standard(user_problem):
    LOGGER.log_with_depth(f"User problem:\n{user_problem}", depth=0)

    system_prompt = f'''
You are an expert in medical. Based on your own knowledge, please solve the user's problem and provide the reason.
First, analyze the user's problem.
Second, explain how you solve the user's problem step by step.
Finally, present the final result of the user's problem in <final_result></final_result>. For a binary problem, it is a value between 0 to 1. You must include the exact probability within the <final_result></final_result> tags, e.g., 'The failure rate of the clinical trial is <final_result>0.8</final_result>.'
Note: You must include the exact probability within the <final_result></final_result> tags.
    '''
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The original user problem is: {user_problem}"},
        {"role": "user", "content": "Please solve the user's problem and provide the reason."}
    ]

    final_results = llm_request(messages)
    LOGGER.log_with_depth(final_results.choices[0].message.content, depth=0)
    LOGGER.log_with_depth("\n======================    END   ================\n", depth=0)

    return None, final_results.choices[0].message.content


if __name__ == "__main__":
    user_prompt = "Predict the DTI of aspirin and ibuprofen on proteins p53 and BRCA1."
    subproblem_solutions, final_solution = solve_problem(user_prompt)
    print("Subproblem Solutions:")
    print(subproblem_solutions)
    print("Final Solution:")
    print(final_solution)