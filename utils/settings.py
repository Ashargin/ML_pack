DATA_PATH = 'data/all-ads-2018.05.01_prepared.csv'
TARGET_COLS = ['CONDOMINIUM_EXPENSES']
DUMMY_VALS = {'FANCY_ZONE': ['AUTRE', 'IDF_HORS_PARIS', 'PARIS'],
             'HEATING_ZONE': ['H1c', 'H1a', 'H2d', 'H3', 'H1b', 'H2c', 'H2b', 'H2a'],
             'REGION': ['AUVERGNE_RHONE_ALPES', 'HAUTS_DE_FRANCE',
                        'PROVENCE_ALPES_COTE_DAZUR', 'GRAND_EST', 'OCCITANIE',
                        'IDF_HORS_PARIS', 'NORMANDIE', 'NOUVELLE_AQUITAINE',
                        'CENTRE_VAL_DE_LOIRE', 'BOURGOGNE_FRANCHE_COMTE',
                        'BRETAGNE', 'PAYS_DE_LA_LOIRE', 'PARIS']}
