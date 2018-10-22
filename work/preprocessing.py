dummyVals = {'FANCY_ZONE': ['AUTRE', 'IDF_HORS_PARIS', 'PARIS'],
             'HEATING_ZONE': ['H1c', 'H1a', 'H2d', 'H3', 'H1b', 'H2c', 'H2b', 'H2a'],
             'REGION': ['AUVERGNE_RHONE_ALPES', 'HAUTS_DE_FRANCE',
                        'PROVENCE_ALPES_COTE_DAZUR', 'GRAND_EST', 'OCCITANIE',
                        'IDF_HORS_PARIS', 'NORMANDIE', 'NOUVELLE_AQUITAINE',
                        'CENTRE_VAL_DE_LOIRE', 'BOURGOGNE_FRANCHE_COMTE',
                        'BRETAGNE', 'PAYS_DE_LA_LOIRE', 'PARIS']}


def preproc_numeric(data):
    data_temp = data.reset_index(drop=True)
    X = data_temp[
        ['CARETAKER', 'ELEVATOR', 'PARKING', 'SURFACE', 'MAIN_CITY', 'HEATING_MODE', 'CONSTRUCTION_YEAR', 'FLOOR',
         'FLOOR_COUNT', 'LOT_COUNT']].copy()
    for var in ['FANCY_ZONE', 'HEATING_ZONE', 'REGION']:
        i = 0
        for val in dummyVals[var]:
            if i > 0:
                X[var + '_' + val] = (data_temp[var] == val)
            i += 1
    return X.set_index(data.index)


def preproc_filled_discrete(data):
    X = preproc_numeric(data)
    X.HEATING_MODE.fillna(-1, inplace=True)
    X.CONSTRUCTION_YEAR.fillna(2050, inplace=True)
    X.FLOOR.fillna(-5, inplace=True)
    X.FLOOR_COUNT.fillna(-5, inplace=True)
    X.LOT_COUNT.fillna(-100, inplace=True)
    return X


def preproc_linreg(data):
    data_temp = data.reset_index(drop=True)
    X = data_temp[
        ['SURFACE', 'CARETAKER', 'HEATING_MODE', 'ELEVATOR', 'FLOOR', 'LOT_COUNT', 'CONSTRUCTION_YEAR', 'PARKING',
         'MAIN_CITY', 'HEATING_ZONE', 'REGION', 'DEPT_CODE', 'ZIP_CODE']].copy()
    X.HEATING_MODE.fillna(0.0, inplace=True)
    X['LOT_COUNT_CAT'] = X.LOT_COUNT.apply(format_lot_count)
    return X.set_index(data.index)


def format_lot_count(x):
    if x <= 0 or x > 500:
        return
    elif x <= 10:
        return '10 lots ou moins'
    elif x <= 49:
        return '11 à 50 lots'
    elif x <= 199:
        return '51 à 200 lots'
    return 'plus de 200 lots'


def format_year(x):
    if x < 1900 or x > 2018:
        return
    elif x <= 1945:
        return '1900-1945'
    elif x <= 1958:
        return '1946-1958'
    elif x <= 1974:
        return '1959-1974'
    elif x <= 2000:
        return '1975-2000'
    return '2001-2018'
