import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


data_path = '/Volumes/michsand256/kait_data/'
graph_path = sys.path[0] + 'graphs'
metrics_path = sys.path[0] + 'metrics'


WN_data_export = 'BALL DATA - UTWN'

FE_data_export = 'BALL DATA - UTFE'

FE_player_list = ['UTFE01', 'UTFE02', 'UTFE03', 'UTFE04', 'UTFE05', 'UTFE06',
                  'UTFE07', 'UTFE08', 'UTFE09', 'UTFE10', 'UTFE11', 'UTFE12']


MN_data_export = 'BALL DATA - UTMN'

MN_player_list = ['UTMN01', 'UTMN03', 'UTMN04', 'UTMN05', 'UTMN06', 'UTMN07',
                  'UTMN08', 'UTMN09', 'UTMN10', 'UTMN11', 'UTMN12']


TR_MAR_22_data_export = 'BALL DATA - TR MAR 2022'

TR_MAR_22_player_list = ['TR01', 'TR02', 'TR04', 'TR05', 'TR07', 'TR09', 'TR10', 'TR11', 'TR12', 'TR14']


TR_SEP_22_data_export = 'BALL DATA - TR SEP 2022'

TR_SEP_22_player_list = ['TR01', 'TR07', 'TR08', 'TR10', 'TR11', 'TR12', 'TR13', 'TR14', 'TR15', 'TR16', 'TR17']


TR_FEB_23_data_export = 'BALL DATA - TR FEB 2023'

TR_FEB_23_player_list = ['TR01', 'TR07', 'TR10', 'TR12', 'TR20', 'TR21']



column_names_alt = ['Frame', 'Time', 'Ball_Position_X', 'Ball_Position_Y', 'Ball_Position_Z',
                    'Rim:Marker001_Position_X', 'Rim:Marker001_Position_Y', 'Rim:Marker001_Position_Z', 'Rim:Marker002_Position_X', 'Rim:Marker002_Position_Y', 'Rim:Marker002_Position_Z',
                    'Rim:Marker003_Position_X', 'Rim:Marker003_Position_Y', 'Rim:Marker003_Position_Z', 'Rim:Marker004_Position_X', 'Rim:Marker004_Position_Y', 'Rim:Marker004_Position_Z']

column_names_alt_2 = ['Frame', 'Time', 'Ball_Position_X', 'Ball_Position_Y', 'Ball_Position_Z',
                      'Rim:Marker001_Position_X', 'Rim:Marker001_Position_Y', 'Rim:Marker001_Position_Z', 'Rim:Marker002_Position_X', 'Rim:Marker002_Position_Y', 'Rim:Marker002_Position_Z',
                      'Rim:Marker003_Position_X', 'Rim:Marker003_Position_Y', 'Rim:Marker003_Position_Z', 'Rim:Marker004_Position_X', 'Rim:Marker004_Position_Y', 'Rim:Marker004_Position_Z',
                      'Rim:Marker005_Position_X', 'Rim:Marker005_Position_Y', 'Rim:Marker005_Position_Z', 'Rim:Marker006_Position_X', 'Rim:Marker006_Position_Y', 'Rim:Marker006_Position_Z']