import pandas as pd
import re

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def string_to_bool(df):
    df.replace({'Sim': True, 'Não': False}, inplace=True)
    return df

def categorize_location(location):
    if type(location) != str:
        return False
    
    location = location.lower()
    if "lobo superior e inferior" in location:
        return "Outros"
    
    elif "língula" in location:
        return "lobo superior esquerdo"
    
    elif("médio" in location):
        return "Lobo médio direito"
    
    elif("direito" in location) or ("direita" in location):
        if "lobo superior" in location or "ápice" in location:
            return "Lobo superior direito"
        elif "lobo inferior" in location or "base" in location or "basal" in location:
            return "Lobo inferior direito"
        else:
            return "Outros"    
        
    elif("esquerda" in location) or ("esquerdo" in location):
        if "lobo superior" in location:
            return "Lobo superior esquerdo"
        elif "lobo inferior" in location or "base" in location or "basal" in location:
            return "Lobo inferior esquerdo"
        else:
            return "Outros"    
    else:
        return "Outros"
    
def extract_size(text):
    if text == False:
        return None
    if type(text) == float:
        text = str(text)
    text = text.lower()
    composed_match = re.search(r'(\d+(?:,\d+)?\s?x\s?\d+(?:,\d+)?\s?(?:x\s?\d+(?:,\d+)?)?\s?(?:cm|mm))', text)
    if composed_match:
        return composed_match.group(0)
    simple_match = re.search(r'(\d+(?:,\d+)?\s?(?:cm|mm))', text)
    if simple_match:
        return simple_match.group(0)
    return None

def convert_diameter_to_mm(value):
    if pd.isna(value):
        return value

    if "cm" in value:
        value = value.replace("cm", "").strip()
        value = value.replace(",", ".")  
        if "x" in value:
            dimention = [float(v.strip()) * 10 for v in value.split("x")]
            min_diameter = min(dimention)
            return f"{min_diameter:.1f}"
        else:
            return f"{float(value) * 10:.1f}"
    elif "mm" in value:
        value = value.replace("mm", "").strip()
        value = value.replace(",", ".")  
        if "x" in value:
            dimention = [float(v.strip()) for v in value.split("x")]
            min_diameter = min(dimention)
            return f"{min_diameter:.1f}"
        else:
            return f"{float(value):.1f}"
    else:
        return value

def structured_location(df):
    df['Localização do nódulo'] = df['Localização do nódulo'].apply(categorize_location)
    return df

def structured_size(df):
    df['Tamanho do nódulo'] = df['Tamanho do nódulo'].apply(extract_size)
    return df

def converted_size(df):
    df['Tamanho do nódulo'] = df['Tamanho do nódulo'].apply(convert_diameter_to_mm)
    return df

if __name__ == '__main__':
    #Zero Shot
    #input_filename = "data/one_lung_nodule/zero_shot/results_gemini/results_prompt_2_structured_v2.csv"
    #output_filename = "data/one_lung_nodule/zero_shot/results_gemini/results_prompt_2_structured_post_processing_v2.csv"

    #Few Shot

    #Gemini 
    input_filename = "data/one_lung_nodule/few_shot/results_gemini/results_prompt_3_one_ex_structured_v2.csv"
    output_filename = "data/one_lung_nodule/few_shot/results_gemini/results_prompt_3_one_ex_structured_post_processing_v2.csv"

    # GPT-4-o
    #input_filename = "data/one_lung_nodule/few_shot/results_gpt4o/results_prompt_2_two_ex_structured.csv"
    #output_filename = "data/one_lung_nodule/few_shot/results_gpt4o/results_prompt_2_two_ex_structured_post_processing_test.csv"

    # Llama 3
    #input_filename = "data/one_lung_nodule/few_shot/results_llama3/results_prompt_3_one_ex_structured.csv"
    #output_filename = "data/one_lung_nodule/few_shot/results_llama3/results_prompt_3_one_ex_structured_post_processing_test.csv"

    df = read_csv(input_filename)
    df = string_to_bool(df)
    df = structured_location(df)
    df = structured_size(df)
    df = converted_size(df)
    df = df.rename(columns={"Tamanho do nódulo": "Tamanho do nódulo (mm)"})

    df.to_csv(output_filename, index=False)

    