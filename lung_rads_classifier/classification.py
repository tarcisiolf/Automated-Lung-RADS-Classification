import pandas as pd

from lungrads import nodule
from lungrads import lung_rads_classifier


def determine_attenuation(row):
    if row['O nódulo é sólido ou em partes moles?']:
        return 'Sólido'
    elif row['O nódulo é semissólido ou subsólido?']:
        return 'Parcialmente Sólido'
    elif row['O nódulo é em vidro fosco?']:
        return 'Vidro Fosco'
    return 'Desconhecido'

def determine_edges(row):
    return 'Espiculada' if row['O nódulo é espiculado, irregular ou mal definido?'] else 'Não Espiculada'

def classify_nodules(df):
    nodules_classification = []
    for _, row in df.iterrows():
        attenuation = determine_attenuation(row)
        edges = determine_edges(row)
        calcification = row['O nódulo é calcificado?']
        location = row['Localização do nódulo']
        if row['Tamanho do nódulo (mm)'] != "False":
            diameter = float(row['Tamanho do nódulo (mm)'])
        
        single_nodule = nodule.Nodule(attenuation=attenuation, edges=edges, calcification=calcification, localization=location, size=diameter, solid_component_size=diameter)
        
        classifier = lung_rads_classifier.LungRADSClassifier(single_nodule)
        results = classifier.classifier()
        nodules_classification.append(results)
    
    return nodules_classification
