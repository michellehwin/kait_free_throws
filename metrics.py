import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
import traceback 
import time
import src.utils as utils


import math
from statistics import mean
from statistics import stdev

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import *
import globals



def main():

    start_time = time.time()

    # -------------------- 1 Player ------------------------

    # data_export = globals.FE_data_export
    # player_num = 0
    # player = globals.FE_player_list[player_num]

    # player_files = [x for x in os.listdir(f'{globals.data_path}/{data_export}/{data_export}/{player}') if 'FT' in x]
    # print(player_files)

    # rows = []
    # for f in player_files:
        
    #     row = metrics(f'{globals.data_path}/{data_export}/{data_export}/{player}/{f}')
    #     rows.append(row)

    # df_final = pd.DataFrame(rows)

    # df_final.to_csv(f'{globals.metrics_path}/{player}/{player}_metrics.csv', index=False)


    # -------------------- All Players ------------------------

    in_folder = '/Volumes/michsand256/kait_data/BALL DATA - UTFE'
    player_list = []
    for folder in os.listdir(in_folder):
        if os.path.isdir(f'{in_folder}/{folder}'):
            player_list.append(folder)
    
    df_list = []

    for i in range (len(player_list)):
        player = player_list[i]
        print('player',player)
        player_files = []
        for file in os.listdir(f'{in_folder}/{player}'):
            if 'FT' in file and '.csv' in file and file[:2] != '._':
                player_files.append(f"{in_folder}/{player}/{file}")

        rows = []
        for f in player_files:

            print(f'-------- {f} --------')
            
            row = utils.metrics('out', f)
            rows.append(row)

        if len(rows) == 0:
            continue
        df_metrics = pd.DataFrame(rows)

        os.makedirs(f'metrics/{player}', exist_ok=True)
        df_metrics.to_csv(f'metrics/{player}/{player}_metrics.csv', index=False)

        df_list.append(df_metrics)

    df_final = pd.concat(df_list)
    df_final.to_csv(f'metrics/{player_list[0][0:4]}_metrics.csv', index=False)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
