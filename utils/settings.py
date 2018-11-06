DATA_PATH = 'data/all-ads-2018.05.01_prepared.csv'
VAL_DATA_PATH = 'data/val_data.csv'
TEST_DATA_PATH = None

TRAIN_PREPROC_PATH = 'data/train_preproc.csv'
VAL_PREPROC_PATH = 'data/val_preproc.csv'
TEST_PREPROC_PATH = 'data/test_preproc.csv'

MEDIANS_PATH = 'data/medians.pkl'
STDS_PATH = 'data/stds.pkl'

TARGET_COL = 'CONDOMINIUM_EXPENSES'
SUB_PRED_VAR = 'LOG_EXPENSES_M2'
DUMMY_VALS = {'FANCY_ZONE': ['AUTRE', 'IDF_HORS_PARIS', 'PARIS'],
              'HEATING_ZONE': ['H1c', 'H1a', 'H2d', 'H3', 'H1b', 'H2c', 'H2b', 'H2a'],
              'REGION': ['AUVERGNE_RHONE_ALPES', 'HAUTS_DE_FRANCE',
                         'PROVENCE_ALPES_COTE_DAZUR', 'GRAND_EST', 'OCCITANIE',
                         'IDF_HORS_PARIS', 'NORMANDIE', 'NOUVELLE_AQUITAINE',
                         'CENTRE_VAL_DE_LOIRE', 'BOURGOGNE_FRANCHE_COMTE',
                         'BRETAGNE', 'PAYS_DE_LA_LOIRE', 'PARIS']}
