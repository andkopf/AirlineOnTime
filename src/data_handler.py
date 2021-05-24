import pandas as pd
import os


def load_data(data_folder):
    files = os.listdir(data_folder)
    files = [x for x in files if x.split('.')[-1] == 'bz2']
    data = list()
    for f in files:
        data.append(pd.read_table(os.path.join(data_folder, f), compression='bz2', sep=',', encoding='ISO-8859-1'))
    
    # load meta data
    plane_data = pd.read_csv(os.path.join(data_folder, 'plane-data.csv'))
    carriers = pd.read_csv(os.path.join(data_folder, 'carriers.csv'))
    airports = pd.read_csv(os.path.join(data_folder, 'airports.csv'))
    
    return(pd.concat(data), plane_data, carriers, airports)
    