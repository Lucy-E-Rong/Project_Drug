import re
import json

from llm import LLMAgent
from affinity_agent import AffinityAgent
from utils import LOGGER
from sub_agent import main as sub_agent_main  # 导入 sub_agent 中的 main 函数
from sub_agent import decomposition, extract_drugs_and_proteins, convert_to_smiles, convert_to_amino_acid_sequence

class MedicalAgent(LLMAgent):
    def __init__(self, user_prompt, depth=1):
        self.user_prompt = user_prompt

        self.name = "medical agent"
        self.role = '''
You are an expert in medical.
'''
        self.tools = [
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
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def affinity_agent(self, drug_smiles, protein_sequence):
        # 日志记录用户提示
        LOGGER.log_with_depth(f"Initial user prompt: {self.user_prompt}", depth=1)
        LOGGER.log_with_depth("Affinity Agent... Initializing", depth=1)
        
        # Step 1: Decompose the original problem into subproblems
        LOGGER.log_with_depth("Planning...", depth=1)
        LOGGER.log_with_depth("[Thought] Least to Most Reasoning: Decompose the original problem", depth=1)
        decomposition_result = decomposition(self.user_prompt, self.tools)
        LOGGER.log_with_depth(f"Decomposition Result: {decomposition_result}", depth=1)

        # Step 2: Extract drugs and proteins from the subproblems
        drugs_tagged, proteins_tagged = extract_drugs_and_proteins(decomposition_result)
        LOGGER.log_with_depth(f"Extracted Drugs Tagged: {drugs_tagged}", depth=1)
        LOGGER.log_with_depth(f"Extracted Proteins Tagged: {proteins_tagged}", depth=1)
        
        # Extract individual drugs and proteins from tagged strings
        drugs = [tag.split('</drug>')[0] for tag in drugs_tagged.split('<drug>') if '</drug>' in tag]
        proteins = [tag.split('</protein>')[0] for tag in proteins_tagged.split('<protein>') if '</protein>' in tag]
        LOGGER.log_with_depth(f"Extracted Drugs: {drugs}", depth=1)
        LOGGER.log_with_depth(f"Extracted Proteins: {proteins}", depth=1)

        # Step 3: Convert each drug to its SMILES notation
        LOGGER.log_with_depth("Converting drugs to SMILES notation...", depth=1)
        drugs_with_smiles = {drug: convert_to_smiles(drug) for drug in drugs}
        LOGGER.log_with_depth(f"Drugs with SMILES: {drugs_with_smiles}", depth=1)

        # Step 4: Convert each protein to its amino acid sequence
        LOGGER.log_with_depth("Converting proteins to amino acid sequences...", depth=1)
        proteins_with_sequences = {protein: convert_to_amino_acid_sequence(protein) for protein in proteins}
        LOGGER.log_with_depth(f"Proteins with Sequences: {proteins_with_sequences}", depth=1)

        # Process each drug-protein pair to predict affinity
        LOGGER.log_with_depth("[Action] Solving each drug-protein affinity problem...", depth=1)
        problem_results = []
        for drug, smiles in drugs_with_smiles.items():
            for protein, sequence in proteins_with_sequences.items():
                LOGGER.log_with_depth(f"Processing drug: {drug} (SMILES: {smiles}), protein: {protein} (Sequence: {sequence})", depth=1)
                affinity_agent_ins = AffinityAgent(depth=2)
                smiles_clean = smiles.replace("<smiles>", "").replace("</smiles>", "")
                sequence_clean = sequence.replace("<sequence>", "").replace("</sequence>", "")
                #response = affinity_agent_ins.DTI_pred(smiles_clean, sequence_clean)
               
               
               
                model_result = affinity_agent_ins.DTI_pred(smiles_clean, sequence_clean)
            # 使用 decomposition 获取答案
                response = decomposition(f"How can I predict the affinity of the drug {smiles} on the disease {sequence}?", tools=affinity_agent_ins.tools)
                processing_drug = f"Processing drug: {drug} (SMILES: {smiles}), protein: {protein} (Sequence: {sequence})"
                # 合并结果并记录日志
                combined_result = f"Processing: {processing_drug}\n, Answer: {response}\n, Model Prediction: {model_result}\n"

                #response =  decomposition(f"How can I predict the affinity of the drug {smiles} on the disease {sequence}?",tools=affinity_agent_ins.tools)
                if response == "":
                    LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
                    problem_results.append("No solution found")
                else:
                    LOGGER.log_with_depth(f"<solution>{combined_result}</solution>", depth=1)
                    problem_results.append(combined_result)
            
        return '\n'.join(problem_results)


def main(original_problem, tools=None):
    # Create an instance of MedicalAgent with the original problem
    medical_agent = MedicalAgent(original_problem)
    # Using one of the tools to simulate drug and protein information, replace these with actual calls
    drug_smiles = "C1=CC=CC=C1"  # Example SMILES
    protein_sequence = "MADSEQ"  # Example sequence

    # Get the affinity result
    result = medical_agent.affinity_agent(drug_smiles, protein_sequence)
    print(result)  # Print the final combined result




    
if __name__ == "__main__":
    user_prompt = "Predict the DTI of aspirin and ibuprofen on proteins p53 and BRCA1."
    agent = MedicalAgent(user_prompt)
    result = agent.affinity_agent("C1=CC=CC=C1", "MADSEQ")
    print(result)

