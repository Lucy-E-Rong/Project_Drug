from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import data_process
from DeepPurpose import utils
from llm import LLMAgent
import os

os.chdir('./save_folder/pretrained_models/model_seed_5/')

path = utils.load_dict('')
net=models.download_pretrained_model_S3('')
model = models.model_initialize(**path) 

'''
new_data = ['O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl']  # 添加您的新分子
new_encoded = data_process(X_drug=new_data, y=[0.7], drug_encoding='rdkit_2d_normalized', split_method='no_split')

# 进行预测
y_pred = model.predict(new_encoded)

# 打印预测结果
print("预测结果:", y_pred)
'''


class AdmetAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "admet agent"
        self.role = ''' 
As an ADMET prediction expert, you have the capability to evaluate the absorption, distribution, metabolism, excretion, and toxicity of a given drug. 
You utilize a deep learning model that can parse the drug's SMILES representation to make predictions of ADMET.
Given the drug SMILES, use a pretrained model to predict the ADMET properties of the drug. 
This includes predictions related to absorption, distribution, metabolism, excretion, and toxicity. 
The agent will be triggered by any inquiries regarding ADMET-related aspects of the drug.
'''

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "ADMET_pred",
                    "description": '''
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

        super().__init__(self.name, self.role, tools=self.tools)
        
    def admet_pred(self,drug):
        x_pred = data_process(X_drug=[drug], y=[0.7], drug_encoding='rdkit_2d_normalized', split_method='no_split')
        y_pred = model.predict(x_pred)
        return y_pred


def test_admet_agent():
    admet_agent = AdmetAgent()

    # 示例药物 SMILES 和目标氨基酸序列
    drug_smiles = "CC(C(=O)O)N"  # 这是一个示例，您可以替换为实际的 SMILES

    # 调用 DTI_pred 方法
    predicted_admet = admet_agent.admet_pred(drug_smiles)

    # 打印结果
    print(f"Predicted admet for the drug smiles:")
    print(predicted_admet)

# 调用测试函数
if __name__ == '__main__':
    test_admet_agent()
