from llm import LLMAgent
import json

from utils import LOGGER

# 1. Least to Most Reasoning
def decomposition(original_problem, tools=None):
    name = 'decomposition agent'
    
    role = '''
As a decomposition expert, you have the capability to break down a complex problem into smaller, more manageable subproblems. 
Utilize tools to address each subproblem individually, ensuring one tool per subproblem.
Aim to resolve every subproblem either through a specific tool or your expertise.
You don't need to solve it; your duty is merely to break down the problem into <subproblem>subproblems</subproblem>.
'''

    examples = '''
Question: How can we predict whether this clinical trial can pass?
Answer: <subproblem> To understand the drug's and disease's safety, we need to consult the Safety Agent for information on drug risks and disease risks. </subproblem>
<subproblem> To evaluate the drug's efficacy against diseases, it is essential to request information on the drug's effectiveness from the Efficacy Agent. Obtain details about the drug, including its description, pharmacological indications, absorption, volume of distribution, metabolism, route of elimination, and toxicity. </subproblem>
<subproblem> Ask the enrollment agent to assess the difficulty of enrolling enough patients. </subproblem>

Question: How can I evaluate the safety of the drug aspirin on the disease diabetes?
<subproblem> To assess the risk associated with the drug, we must use the "get_drug_risk" tool. </subproblem>
<subproblem> To evaluate the risk associated with the disease, we must use the "get_disease_risk" tool. </subproblem>
<subproblem> Provide insights into the safety of both the drug and the disease based on your expertise, without resorting to any external tools. </subproblem>

Question: How can I evaluate the efficacy of the drug aspirin on the disease diabetes?
Answer:
<subproblem> To understand the drug's structure, obtaining the SMILES notation of the drug is necessary. </subproblem>
<subproblem> Assessing the drug's effectiveness requires retrieving information from the DrugBank database. </subproblem>
<subproblem> To evaluate the drug's impact on the disease, we must obtain the pathway linking the drug to the disease from the Hetionet Knowledge Graph by using the retrieval_hetionet tool. </subproblem>
<subproblem> Offer insights on the drug and disease based on your expertise, without resorting to any external tools. </subproblem>
'''

    if tools and len(tools) > 0:
        func_content = json.dumps([
            {'function_name': func['function']['name'], 'description': func['function']['description']} for func in tools
        ], indent=4)

        role += f"\n The following tools are available for you to use: <tools>{func_content}</tools>."

    agent1 = LLMAgent(name, role, examples=examples)

    response = agent1.request(original_problem)

    return response

# 2. 提取药物和蛋白
def extract_drugs_and_proteins(sub_problem):
    name = 'extraction agent'
    
    role = '''
As an extraction expert, your task is to identify and extract the names of drugs and proteins from a given problem.
Provide two separate lists: one for drugs and one for proteins.
Ensure accuracy and avoid any irrelevant terms. do not provide extra content.
'''

    examples = '''
Question: Identify drugs and proteins in the following problem: "Evaluate the safety of aspirin and ibuprofen on proteins p53 and BRCA1."
{
    "Drugs": ["aspirin", "ibuprofen"],
    "Proteins": ["p53", "BRCA1"]
}

Question: What are the drugs and proteins in this query: "Analyze the interaction between acetaminophen and protein kinase A."
{
    "Drugs": ["acetaminophen"],
    "Proteins": ["protein kinase A"]
}
'''

    agent2 = LLMAgent(name, role, examples=examples)

    response = agent2.request(sub_problem)
    
    # Extract drugs and proteins from the response (expected to be in JSON format or similar)
    data = json.loads(response)
    drugs = data.get('Drugs', [])
    proteins = data.get('Proteins', [])

    # Add identifying tags to the response
    drugs_tagged = ''.join([f'<drug>{drug}</drug>' for drug in drugs])
    proteins_tagged = ''.join([f'<protein>{protein}</protein>' for protein in proteins])

    return drugs_tagged, proteins_tagged

# 3. 药物转换成smiles
def convert_to_smiles(drug_name):
    name = 'smiles conversion agent'
    
    role = '''
As an expert in molecular representation, your task is to convert drug names into SMILES strings.
Ensure accuracy and provide the standardized SMILES notation for each drug name. do not provide extra content.
'''

    examples = '''
Question: Convert the drug names to SMILES strings: "aspirin, ibuprofen."
CC(=O)OC1=CC=CC=C1C(=O)O
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O

Question: What are the SMILES for "paracetamol, morphine?"
CC(=O)NC1=CC=C(O)C=C1
CN1CCC23C4CC=C5[C@H]2[C@H]2[C@H](O)C=C[C@]21[C@H]5CC[C@]34O 
'''

    agent3 = LLMAgent(name, role, examples=examples)

    response = agent3.request(drug_name)

    tagged_response = f'<smiles>{response.strip()}</smiles>'
    
    return tagged_response

# 4. 转换蛋白质为氨基酸序列
def convert_to_amino_acid_sequence(protein_name):
    name = 'amino acid sequence conversion agent'
    
    role = '''
As an expert in protein structures, your task is to convert protein names to their amino acid sequences.
Ensure that the sequences are accurate and standardized for each protein name. do not provide extra content.
'''

    examples = '''
Question: Convert the protein names to amino acid sequences: "p53, BRCA1."
MEEPQSDPSVEPPLSQETFSDLWKLLPEN
MQLAAIALGSPADVILCKLIFYSKGQFLG

Question: What are the sequences for the proteins "tyrosine kinase, protein kinase A?"
MASSESTSSSDGSDQASTQLTPTFPQDGGRVKKNAGYGLN
MGTEKGESSGAGASGSGSNSSIHAEVS
'''

    agent4 = LLMAgent(name, role, examples=examples)

    response = agent4.request(protein_name)

    tagged_response = f'<sequence>{response.strip()}</sequence>'
    
    return tagged_response

def main(original_problem, tools=None):
    # Step 1: Decompose the original problem into subproblems
    decomposition_result = decomposition(original_problem, tools)

    # Step 2: Extract drugs and proteins from the subproblem
    drugs_tagged, proteins_tagged = extract_drugs_and_proteins(decomposition_result)
    
    # Extract individual drugs and proteins from tagged strings
    drugs = [tag.split('</drug>')[0] for tag in drugs_tagged.split('<drug>') if '</drug>' in tag]
    proteins = [tag.split('</protein>')[0] for tag in proteins_tagged.split('<protein>') if '</protein>' in tag]

    # Step 3: Convert each drug to its SMILES notation
    drugs_with_smiles = {drug: convert_to_smiles(drug) for drug in drugs}

    # Step 4: Convert each protein to its amino acid sequence
    proteins_with_sequences = {protein: convert_to_amino_acid_sequence(protein) for protein in proteins}

    combined_drugs = ''.join([f"<drug>{drug}</drug> {drugs_with_smiles[drug]}\n" for drug in drugs_with_smiles])
    combined_proteins = ''.join([f"<protein>{protein}</protein> {proteins_with_sequences[protein]}\n" for protein in proteins_with_sequences])

    return {
        'decomposition': decomposition_result,
        'drugs': combined_drugs.strip(),
        'proteins': combined_proteins.strip()
    }

if __name__ == "__main__":
    original_problem = "Evaluate the safety of aspirin and ibuprofen on proteins p53 and BRCA1."
    tools = None  # Define tools if there are any

    result = main(original_problem, tools)
    print(json.dumps(result, indent=4))