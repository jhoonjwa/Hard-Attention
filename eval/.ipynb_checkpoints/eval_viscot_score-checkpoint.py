import os
import time
import numpy as np
from tqdm import tqdm
import json
import re
from joblib import Parallel, delayed
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

BASE_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: %s
standard answer: %s
model's answer: %s
"""

score_keys = ['output_score', 'output_with_1.0_True_1.0_options_False_score', 'output_with_1.5_True_1.0_options_False_score', 
              'output_with_5.0_True_1.0_options_False_score', 'output_with_100.0_True_1.0_options_False_score', 
              'output_with_1000.0_True_1.0_options_False_score', 'output_with_1.5_True_3.0_options_False_score', 
              'output_with_5.0_True_3.0_options_False_score', 'output_with_100.0_True_3.0_options_False_score', 
              'output_with_1000.0_True_3.0_options_False_score', 'output_with_1.5_True_10.0_options_False_score', 
              'output_with_5.0_True_10.0_options_False_score', 'output_with_100.0_True_10.0_options_False_score', 
              'output_with_1000.0_True_10.0_options_False_score', 'output_with_1.5_True_100.0_options_False_score', 
              'output_with_5.0_True_100.0_options_False_score', 'output_with_100.0_True_100.0_options_False_score']



def make_request_openai(content):
    try:
        os.environ['OPENAI_API_KEY'] = 'sk-None-UNJfMHuxjgqNv5H0IYvkT3BlbkFJbXxSNAe7WFviK13vtGmn'
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": BASE_PROMPT},
                {"role": "user", "content": content}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return str(e)

def get_keys(dataset_result):
    key_list = []
    keys = dataset_result.keys()

    key_list = [key for key in keys if key.split('_')[0] == 'output']
    return key_list


def get_score(dataset_result):
    if isinstance(dataset_result, str):
        return
    #keys = ["output", "output_with_1.5_True_1.0", "output_with_3.0_True_1.0", "output_with_5.0_True_1.0", "output_with_5.0_True_2.0", "output_with_3.0_False_1.0", "output_with_1.0_True_2.0"]
    keys = get_keys(dataset_result)
    for target_key in keys:        
        if target_key + '_score' in dataset_result.keys() or target_key in score_keys:
            return dataset_result
        question, answer, output = dataset_result['question'], dataset_result['answer'], dataset_result[target_key]
        content = PROMPT % (question, answer, output)
        ret = make_request_openai(content)
        ret = ret.lower()
        print(ret)

        if 'score' not in ret:
            return ret
        
        res = re.findall(r'score: ([\d\.]+)', ret)
        if len(res) != 1:
            return 0.0
        res = float(res[0])
        if res > 1.0:
            res = 1
        if res < 0.0:
            res = 0
        dataset_result[target_key + '_score'] = res
       
    return dataset_result



if __name__ == "__main__":
    with open('/home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json', 'r') as f:
        dataset_result = json.load(f)
    with open('/home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_annotated.json', 'r') as f:
        original = json.load(f)
    #dataset_results = dataset_results[:12]
    BATCH = 16
    dataset_results_evaluated = []
    count = 0
    
    data_list = [data['bbox'] for data in original]
    
    dataset_results = []
    for idx, data in enumerate(dataset_result):
        if data['bbox'] not in data_list:
            dataset_results.append(dataset_result[idx])
    

    dataset_results = dataset_result     
    print(len(dataset_results))
    
    for i in range(0, len(dataset_results), BATCH):
        print(f"Processing {i} to {min(i+BATCH, len(dataset_results))}")
        outputs = Parallel(n_jobs=BATCH, verbose=100, prefer="threads")(delayed(get_score)(dataset_result) for idx, dataset_result in enumerate(dataset_results[i:min(i+BATCH, len(dataset_results))]))
        for output in outputs:
            dataset_results_evaluated.append(output)
        
        print(f"Finished {i} to {min(i+BATCH, len(dataset_results))}")
        
    
    data_to_be_saved = [data for data in dataset_results_evaluated if isinstance(data, dict)]

    with open('/home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_annotated.json', 'w') as f:
        json.dump(data_to_be_saved, f, indent=4)