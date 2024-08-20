'''
pip install gradio
pip install rdkit
pip install rdkit-pypi
pip install git+https://github.com/bp-kelley/descriptastorus 
pip install pandas-flavor
'''

import os
import sys
import numpy as np
import pandas as pd
from tools.DP.DeepPurpose import utils
from tools.DP.DeepPurpose import DTI as models
import gradio as gr


sys.path.append(os.getcwd())
from main import solve_problem

model = models.model_pretrained(model = 'MPNN_CNN_BindingDB')

def DTI_pred(drug, target):
    X_pred = utils.data_process(X_drug = [drug], X_target = [target], y = [0.7],
    drug_encoding = 'MPNN', target_encoding = 'CNN',
    split_method='no_split')
    y_pred = model.predict(X_pred)

    return str(y_pred[0])

# 处理LLM问题的函数
def process_llm(question):
    # 调用 solve_problem 功能
    llm_answer = solve_problem(question)
    return llm_answer

# 更新后的处理函数
def combined_function(question, drug_smiles, target_sequence):
    affinity_score = DTI_pred(drug_smiles, target_sequence)
    llm_answer = process_llm(question)
    return llm_answer, affinity_score

interface = gr.Interface(
    fn=combined_function,
    inputs=[
        gr.inputs.Textbox(lines=5, label="Input your question"),
        gr.inputs.Textbox(lines=5, label="Drug SMILES"),
        gr.inputs.Textbox(lines=5, label="Target Amino Acid Sequence")
    ],
    outputs=[
        gr.outputs.Textbox(label="LLM Answer"),
        gr.outputs.Textbox(label="Predicted Affinity")
      ]
)

interface.launch(share=True)