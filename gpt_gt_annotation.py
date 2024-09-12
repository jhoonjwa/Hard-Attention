from openai import OpenAI
from joblib import Parallel, delayed
from dotenv import load_dotenv
import json




client = OpenAI(
    api_key='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',
)
input_path = '/home/vqa/data/outputs/viscot/llava_v15_7b_4.json'

with open(input_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)



def run_model(datum, prompt):    
    print(f"Datum index: {datum['index']}")
    response = client.chat.completions.create(
        model= "gpt-4",
        response_format = { "type": "json_object" },
        messages= [
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature= 0,
        max_tokens= 1024
    )
    output = response.choices[0].message.content

    return {
        **datum,
        "output": output
    }

prompt = '''You are a question answering expert. Your job is to assist the user to find what should the user focus in the given question. 
            To assist the user, you must give specific part of the question in raw form. You should not rephrase it in any form

            Here are examples of "denied" assistance that you should avoid. Keep in mind of the reasons why it was denied.:
            Question: Can you describe the special offer mentioned on the sign at the market?
            Tokenized_Question: ['Can', '▁you', '▁describe', '▁the', '▁special', '▁offer', '▁mentioned', '▁on', '▁sign', '▁at', '▁market', '?']
            { "Guidance": ['offer in the market'] } (denied - you should not rephrase the word in the sentence. Answer ["_offer", "_market" ] which is the raw form in the sentence)
            
            Question: Where is the man pointing at?
            { Guidance: [fingertips] } (denied - you should never give parts that are not present in the question)

            Question: According to the document, when will the person leave the room?
            Guidance: Document, Leave (denied - keep the exact lower/uppercase for each word)

            Instead you should answer it like this:
            Here are the "accepted" versions of the "denied" assistance.

            Question: What is the color of the man's hat?
            { Guidance: [color, man, hat] } (accepted - raw form used)

            Question: Where is the man pointing at?
            { Guidance: [man] } (exist in the question)

            Question: According to the document, when will the person leave the room?
            { Guidance: [document, leave] } (denied - keep the exact lower/uppercase for each word)




            '''