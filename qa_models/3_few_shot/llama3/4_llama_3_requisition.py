import time
import tqdm
import os
from together import Together
import sys
sys.path.append(r'src\utils')
import process_files as pf

def llama_req(inputs, prompts):
    results = []
    total_cost = 0
    total_exec_time = 0
    total_tokens = 0

    for i in range(len(inputs)):
        
        prompt = prompts[i]
        input = inputs[i]
        together_key = os.environ.get('TOGETHER_API_KEY')
        client = Together(api_key=together_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input}
            ],
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
    inputs = pf.read_input_file(r"data\one_lung_nodule\inputs.txt")
    prompts = pf.read_input_file(r"data\one_lung_nodule\few_shot\prompt_3_two_ex.txt")

    inputs = pf.pre_process_input_file(inputs)
    prompts = pf.pre_process_input_file(prompts)

    results = llama_req(inputs, prompts)
    pf.write_output_file(r"data\one_lung_nodule\few_shot\results_llama3\results_prompt_3_two_ex.txt", results)