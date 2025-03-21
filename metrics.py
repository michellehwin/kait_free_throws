import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
import traceback 
import time


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

    data_export = globals.MN_data_export
    player_list = globals.MN_player_list 
    
    df_list = []

    for i in range (len(player_list)):

        player = player_list[i]

        player_files = [x for x in os.listdir(f'{globals.data_path}/{data_export}/{data_export}/{player}') if 'FT' in x]
        print(player_files)

        rows = []
        for f in player_files:

            print(f'-------- {f} --------')
            
            row = metrics(data_export, f'{globals.data_path}/{data_export}/{data_export}/{player}/{f}')
            rows.append(row)

        df_metrics = pd.DataFrame(rows)

        df_metrics.to_csv(f'{globals.metrics_path}/{player}/{player}_metrics.csv', index=False)

        df_list.append(df_metrics)

    df_final = pd.concat(df_list)
    df_final.to_csv(f'{globals.metrics_path}/{player_list[0][0:4]}_metrics.csv', index=False)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
