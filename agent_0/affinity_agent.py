from llm import LLMAgent
#from utils import LOGGER

from tools.DP.DeepPurpose import utils
from tools.DP.DeepPurpose import DTI as models

model = models.model_pretrained(model = 'MPNN_CNN_BindingDB')


class AffinityAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "affinity agent"
        self.role = ''' 
As an affinity expert, you have the capability to evaluate the efficacy of a drug by predicting the binding affinity (KD value or IC50 value) between the drug and the target protein.
You utilize a deep learning model that can parse the drug's SMILES representation and the target's amino acid sequence to make predictions.
'''

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "DTI_pred",
                    "description": '''
                    Given the drug SMILES and the target amino acid sequence, use a pretrained model to predict the Drug-Target Interaction (DTI).
                    The model will return the predicted binding affinity value based on the input data.
                    ''',
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug": {
                                "type": "string",
                                "description": "The Drug SMILES",
                            },
                            "target": {
                                "type": "string",
                                "description": "The Target Amino Acid Sequence",
                            }
                        },
                        "required": ["drug", "target"],
                    },
                }
            },
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)


    def DTI_pred(self, drug, target): 
        X_pred = utils.data_process(X_drug=[drug], X_target=[target], y=[0.7],
                                    drug_encoding='MPNN', target_encoding='CNN',
                                    split_method='no_split')
        y_pred = model.predict(X_pred)

        return str(y_pred[0])



def test_affinity_agent():
    # 创建 AffinityAgent 实例
    affinity_agent = AffinityAgent()

    # 示例药物 SMILES 和目标氨基酸序列
    drug_smiles = "CC(C(=O)O)N"  # 这是一个示例，您可以替换为实际的 SMILES
    target_sequence = "MTEITAAMVKELRESTGAGM"  # 这是一个示例，您可以替换为实际的氨基酸序列

    # 调用 DTI_pred 方法
    predicted_affinity = affinity_agent.DTI_pred(drug_smiles, target_sequence)

    # 打印结果
    print(f"Predicted binding affinity (KD/IC50) for the drug-target pair:")
    print(predicted_affinity)

# 调用测试函数
if __name__ == '__main__':
    test_affinity_agent()