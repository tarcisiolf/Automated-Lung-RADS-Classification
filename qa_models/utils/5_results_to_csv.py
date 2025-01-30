import json
import pandas as pd
import sys
sys.path.append('src/utils')
import process_files as pf

def list_json_to_csv(results):
    df = pd.DataFrame(results, columns=['json_response'])
    return df

def create_structured_df(df):
    new_df = pd.DataFrame()
    for idx, row in df.iterrows():
        nodule = idx
        json_string = row.iloc[0]
        print(nodule, json_string)
        json_dict = json.loads(json_string)
        aux_df = pd.json_normalize(json_dict)
        aux_df.insert(0, "Nódulo", nodule)
        new_df = pd.concat([new_df, aux_df])
    
    return new_df

if __name__ == '__main__':

    # Zero Shot
    #input_filename = "data/one_lung_nodule/zero_shot/results_gemini/results_prompt_1_v2.txt"
    #output_filename = "data/one_lung_nodule/zero_shot/results_gemini/results_prompt_1_structured_v2.csv"

    # Few Shot
    input_filename = "data/one_lung_nodule/few_shot/results_gemini/results_prompt_3_two_ex_v2.txt"
    output_filename = "data/one_lung_nodule/few_shot/results_gemini/results_prompt_3_two_ex_structured_v2.csv"

    results = pf.read_input_file_as_list(input_filename)
    df = list_json_to_csv(results)
    new_df = create_structured_df(df)
    new_df.to_csv(output_filename, index=False)

