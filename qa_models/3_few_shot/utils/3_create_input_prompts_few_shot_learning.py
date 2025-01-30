import json
from tqdm import tqdm

def read_data(file_name):
    return json.load(open(file_name, encoding="utf-8"))


def read_idx(file_name):
    print("reading ...")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx


def construct_prompt(train_data, train_tables, test_data, example_idx=None, example_num=1):
    print("prompt ...")

    def get_example(index):
        exampel_prompt = ""
        for idx_ in example_idx[index][:example_num]:
            id = int(train_data[idx_]["id"])
            text = train_data[idx_]["text"]
            table = train_tables[idx_]

            # Ad ID
            exampel_prompt += f"O laudo exemplo: Id ({id}) {text}\n"
            exampel_prompt += f"O laudo exemplo com a tabela preenchida: {table}\n"
        return exampel_prompt
        
    results = []
    inputs = []

    for item_idx in tqdm(range(len(test_data))):

        item_ = test_data[item_idx]
        id = item_["id"]
        text = item_["text"]

        # PROMPT 1
        """
        prompt = Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"Id do laudo" : "",
"O nódulo é sólido?" : "",
"O nódulo é em partes moles, semissólido ou subsólido?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.
Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

"""

        # PROMPT 2
        
        prompt = """Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"Id do laudo" : "",
"O nódulo é sólido?" : "",
"O nódulo é em partes moles, semissólido ou subsólido?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

A seguir são descritos alguns requisitos para extração:
1. Por favor extraia informações estruturadas para o nódulo pulmonar mencionado no laudo para preencher a tabela. Nesse processo você deve desconsiderar todos os achados descritos no laudo exceto: nódulos, imagem ovalar hiperdensa ou imagem hiperatenuante ovalar.
2. Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. 
3. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.
4. Se o laudo contiver mais de um nódulo descrito crie a quantidade de tabelas necessárias para armazenar as informações relevantes de todos os nódulos pulmonares.

Aqui são descritos alguns pontos de conhecimento médico prévio para sua referência
1. Imagem ovalar hiperdensa deve ser considerada como nódulo pulmonar calcificado.
2. Sólido, partes moles e vidro fosco, são mutuamente exclusivas. Apenas uma das três perguntas pode ser "Sim", e a 
opacidade mista em vidro fosco significa que o tumor tem componentes de opacidade sólidos e em vidro fosco.
3. Micronódulo é um nódulo no pulmão com menos de 3 milímetros (mm) de diâmetro. Nesse contexto devido as suas pequenas dimensões não estamos interessados em extrair suas características. Portanto, não deve ser extraída as suas características.
Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

"""
        # PROMPT 3

        """       
        prompt = Por favor extraia informações estruturadas relevantes do laudo acima:
"Questão" : "Resposta"
{
"O nódulo é sólido?" : "",
"O nódulo é em partes moles, semissólido ou subsólido?" : "",
"O nódulo é em vidro fosco?" : "",
"O nódulo é espiculado, irregular ou mal definido?" : "",
"O nódulo é calcificado?" : "",
"Localização do nódulo" : "",
"Tamanho do nódulo" : ""
}

A seguir são descritos alguns requisitos para extração:
1. Por favor extraia informações estruturadas para o nódulo pulmonar mencionado no laudo para preencher a tabela. Nesse processo você deve desconsiderar todos os achados descritos no laudo exceto: nódulos, imagem ovalar hiperdensa ou imagem hiperatenuante ovalar.
2. Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica, por favor preencha com "Não" a resposta dessa pergunta. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.

Aqui são descritos alguns pontos de conhecimento médico prévio para sua referência
1. Imagem ovalar hiperdensa deve ser considerada como nódulo pulmonar calcificado.
2. Sólido, partes moles e vidro fosco, são mutuamente exclusivas. Apenas uma das três perguntas pode ser "Sim", e a 
opacidade mista em vidro fosco significa que o tumor tem componentes de opacidade sólidos e em vidro fosco.
3. Micronódulo é um nódulo no pulmão com menos de 3 milímetros (mm) de diâmetro. Nesse contexto devido as suas pequenas dimensões não estamos interessados em extrair suas características. Portanto, não deve ser extraída as suas características.
Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.

O laudo exemplo: TOMOGRAFIA COMPUTADORIZADA DO TÓRAX TÉCNICA : Exame realizado em equipamento multislice , sem a infusão endovenosa de contraste iodado . RELATÓRIO : Espessamento parietal brônquico difuso , sendo mais acentuadamente nos lobos inferiores . Destaca - se nos segmentos basais do lobo inferior esquerdo , onde se observa opacidade consolidativa de aspecto atelectásico associado a bronquiectasias cilíndricas , acentuado espessamento parietal brônquico e bolhas enfisematosas subpleurais de permeio , determinando elevação da hemicúpula diafragmática deste lado . Moderado / acentuado enfisema predominantemente parasseptal com predomínio nos campos médios e superiores dos pulmões . Nódulo calcificado no segmento anterior do lobo superior direito , medindo 1 , 1 cm , de aspecto benigno / residual . Traqueia e brônquios principais permeáveis e de calibre conservado . Aumento da área cardíaca . Estruturas vasculares do mediastino de calibre habitual , identificando - se calcificações parietais no trajeto da croça , aorta descendente e artérias coronárias . Não se observam linfonodomegalias mediastinais e hilares . Ausência de derrame pleural . Alterações degenerativas na coluna dorsal .
O laudo exemplo com a tabela preenchida: {'O nódulo é sólido?': 'Não', 'O nódulo é em partes moles, semissólido ou subsólido?': 'Não', 'O nódulo é em vidro fosco?': 'Não', 'O nódulo é espiculado, irregular ou mal definido?': 'Não', 'O nódulo é calcificado?': 'Sim', 'Localização do nódulo': 'no segmento anterior do lobo superior direito', 'Tamanho do nódulo': '1,1 cm'}
O laudo exemplo: TOMOGRAFIA COMPUTADORIZADA DO TÓRAX Exame realizado em caráter de urgência Indicação do exame : investigação de pneumonia viral . Técnica : imagens obtidas por aquisição volumétrica multislice , sem a administração intravenosa de contraste iodado . Aspectos observados : Focos de opacidade em vidro fosco periféricos esparsos nos lobos inferiores , mais evidentes nas bases pulmonares . Discretos espessamentos pleurais focais posteriores bilaterais . Dois diminutos nódulos pulmonares não calcificados no lobo superior direito medindo até 0 , 5 x 0 , 4 cm . Restante do parênquima pulmonar com atenuação preservada . Ausência de derrame pleural . Não se observam linfonodomegalias mediastinais . Traqueia e brônquios principais pérvios , com calibre normal . Estruturas vasculares mediastinais com trajeto e diâmetro preservados . Sinais de espondilose dorsal .
O laudo exemplo com a tabela preenchida: {'O nódulo é sólido?': 'Não', 'O nódulo é em partes moles, semissólido ou subsólido?': 'Não', 'O nódulo é em vidro fosco?': 'Não', 'O nódulo é espiculado, irregular ou mal definido?': 'Não', 'O nódulo é calcificado?': 'Não', 'Localização do nódulo': 'no lobo superior direito', 'Tamanho do nódulo': '0,5 x 0,4 cm'}
O laudo exemplo: TOMOGRAFIA COMPUTADORIZADA DO TÓRAX Técnica : Imagens obtidas por aquisição volumétrica multislice , sem a administração intravenosa do meio de contraste iodado . Análise : Múltiplos micronódulos centrolobulares com atenuação em vidro fosco , na periferia dos lobos superiores . Pequeno nódulo pulmonar calcificado , medindo cerca de 0 , 4 cm , localizado no lobo inferior esquerdo , de aspecto residual . Discretas lâminas atelectásicas esparsas bilateralmente . Pequenos linfonodos calcificados no hilo pulmonar esquerdo , residuais . Traqueia e brônquios - fonte pérvios , com calibre normal . Imagem cardíaca de morfologia normal , sem alterações volumétricas apreciáveis . Estruturas vasculares mediastinais com trajeto e diâmetro preservados . Alterações degenerativas vertebrais .
O laudo exemplo com a tabela preenchida: {'O nódulo é sólido?': 'Não', 'O nódulo é em partes moles, semissólido ou subsólido?': 'Não', 'O nódulo é em vidro fosco?': 'Não', 'O nódulo é espiculado, irregular ou mal definido?': 'Não', 'O nódulo é calcificado?': 'Sim', 'Localização do nódulo': 'no lobo inferior esquerdo', 'Tamanho do nódulo': '0,4 cm'}
O laudo exemplo: TOMOGRAFIA COMPUTADORIZADA DO TÓRAX - RELATÓRIO DE EMERGÊNCIA - Indicação : Síndrome gripal e dispneia aos esforços . Método : Aquisição volumétrica , sem contraste . Achados pulmonares : - Ausência de opacidades pulmonares focais sugestivas de comprometimento inflamatório / infeccioso pulmonar em atividade suspeitas para etiologia viral . - Nódulo calcificado no lobo inferior direito , medindo 7 mm e micronódulo calcificado no lobo inferior direito , medindo 2 mm , residuais . - Restante do parênquima pulmonar com atenuação preservada . Demais achados : Não há sinais de cardiomegalia . Não há derrame pericárdico . Ausência de derrame pleural . Traqueia e brônquios principais pérvios e com calibres conservados . Não há linfonodomegalias mediastinais . Linfonodos hilares calcificados à direita , residuais . Grandes vasos do mediastino de trajeto e calibre conservados . Arcabouço ósseo torácico sem particularidades
O laudo exemplo com a tabela preenchida: {'O nódulo é sólido?': 'Não', 'O nódulo é em partes moles, semissólido ou subsólido?': 'Não', 'O nódulo é em vidro fosco?': 'Não', 'O nódulo é espiculado, irregular ou mal definido?': 'Não', 'O nódulo é calcificado?': 'Sim', 'Localização do nódulo': 'no lobo inferior direito', 'Tamanho do nódulo': '7 mm'}
"""


        prompt += get_example(index=item_idx)
        prompt += '\n'

        input = f"Dado o laudo: Id ({id}) {text}\nRetornar a tabela do laudo preenchida no formato JSON:\n\n"

        inputs.append(input)
        results.append(prompt)
    
    return inputs, results


if __name__ == '__main__':
    train_samples = read_data(r"data\all_lung_nodule\few_shot\train_samples.json")
    test_samples = read_data(r"data\all_lung_nodule\few_shot\test_samples.json")
    train_tables = read_data(r"data\all_lung_nodule\few_shot\train_tabels.json")

    example_idx = read_idx(r"data\all_lung_nodule\few_shot\test.100.simcse.dev.5.knn.jsonl")
    inputs, prompts = construct_prompt(train_data=train_samples, train_tables=train_tables, test_data=test_samples, example_idx=example_idx, example_num=5)

    #with open(r"data\all_lung_nodule\inputs.txt", encoding="utf-8", mode="w") as txt_file:
    #    for line in inputs:
    #        txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    
    with open(r"data\all_lung_nodule\few_shot\prompt_2_five_ex.txt", encoding="utf-8", mode="w") as txt_file:
        for line in prompts:
            txt_file.write("".join(line) + "\n") # works with any number of elements in a line