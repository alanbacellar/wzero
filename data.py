from select import select
from datasets.dataset import pre_processing
import numpy as np


def get_pre_processings(data, pre_processings):
    selected_data = []
    for dat in data:
        if pre_processings in data['pre_processings']:
            selected_data.appned(dat)
    return selected_data


get_dict = {
    'pre_processings': get_pre_processings
}

def get(data, **kwargs):
    
    selected_data = data

    for name, value in kwargs.items():
        
        if name not in data:
            raise ValueError("")
        
        if name in get_dict:
            selected_data = get_dict[name](selected_data, value)
        
        else:
            new_selected_data = []
            for data in selected_data:
                if data[name] == value:
                    new_selected_data = data
            selected_data = new_selected_data