import google.generativeai as genai
import os
import time
import sys
sys.path.append('src/utils')
import process_files as pf

def gemini_req(inputs, prompt):
    results = []
    total_cost = 0
    total_exec_time = 0
    total_tokens = 0

    gemini_key = os.environ.get('GEMINI_API_KEY')
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", 
                                                                         "temperature" : 0})

    for i in range(len(inputs)):     
        input = inputs[i]      

        start_time = time.time()      
        
        final_prompt = ''
        final_prompt = input+"\n\n"+prompt

        response = model.generate_content(final_prompt)
        print(response.text)

        ## Tempo 
        end_time = time.time()
        exec_time = end_time - start_time
        total_exec_time += exec_time
        
        pf.print_execution_stats(i, exec_time, total_exec_time)
        results.append(response.text)
        
        # Sleep to avoid hitting the rate limit
        time.sleep(5)
    return results


if __name__ == '__main__':
    
    #1
    inputs = pf.read_input_file("data/one_lung_nodule/inputs.txt")
    prompt = pf.read_prompt_file("data/one_lung_nodule/zero_shot/prompt_1.txt")

    inputs = pf.pre_process_input_file(inputs)

    results = gemini_req(inputs, prompt)
    pf.write_output_file("data/one_lung_nodule/zero_shot/results_gemini/results_prompt_1_v2.txt", results)

    #2
    inputs = pf.read_input_file("data/one_lung_nodule/inputs.txt")
    prompt = pf.read_prompt_file("data/one_lung_nodule/zero_shot/prompt_2.txt")

    inputs = pf.pre_process_input_file(inputs)

    results = gemini_req(inputs, prompt)
    pf.write_output_file("data/one_lung_nodule/zero_shot/results_gemini/results_prompt_2_v2.txt", results)

    
