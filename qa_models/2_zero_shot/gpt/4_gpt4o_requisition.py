import time
import tqdm
import os
from openai import OpenAI
import sys
sys.path.append(r'src\utils')
import process_files as pf

def gpt_req(inputs, prompt):
    results = []
    total_cost = 0
    total_exec_time = 0
    total_tokens = 0

    for i in range(len(inputs)):
        input = inputs[i]
        openai_key = os.environ.get('OPENAI_API_KEY')
        client = OpenAI(api_key=openai_key)
        

        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input}
            ],
            seed=42,
            temperature=0
        )

        ## Tempo 
        end_time = time.time()
        exec_time = end_time - start_time
        total_exec_time += exec_time
        
        pf.print_execution_stats(i, exec_time, total_exec_time)
        results.append(response.choices[0].message.content)

    return results
    

if __name__ == '__main__':
    inputs = pf.read_input_file(r"data\all_lung_nodule\inputs.txt")
    prompt = pf.read_prompt_file(r"data\all_lung_nodule\zero_shot\prompt_2.txt")

    inputs = pf.pre_process_input_file(inputs)

    results = gpt_req(inputs, prompt)
    pf.write_output_file(r"data\all_lung_nodule\zero_shot\results_gpt4o\results_prompt_2.txt", results)