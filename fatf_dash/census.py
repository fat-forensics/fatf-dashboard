map_i_s = {
    'workclass': {0: 'Federal-gov',
                  1: 'Local-gov',
                  2: 'Private',
                  3: 'Self-emp-inc',
                  4: 'Self-emp-not-inc',
                  5: 'State-gov',
                  6: 'Without-pay'},
    'education': {0: '10th',
                  1: '11th',
                  2: '12th',
                  3: '1st-4th',
                  4: '5th-6th',
                  5: '7th-8th',
                  6: '9th',
                  7: 'Assoc-acdm',
                  8: 'Assoc-voc',
                  9: 'Bachelors',
                  10: 'Doctorate',
                  11: 'HS-grad',
                  12: 'Masters',
                  13: 'Preschool',
                  14: 'Prof-school',
                  15: 'Some-college'},
    'marital-status': {0: 'Divorced',
                       1: 'Married-AF-spouse',
                       2: 'Married-civ-spouse',
                       3: 'Married-spouse-absent',
                       4: 'Never-married',
                       5: 'Separated',
                       6: 'Widowed'},
    'occupation': {0: 'Adm-clerical',
                   1: 'Armed-Forces',
                   2: 'Craft-repair',
                   3: 'Exec-managerial',
                   4: 'Farming-fishing',
                   5: 'Handlers-cleaners',
                   6: 'Machine-op-inspct',
                   7: 'Other-service',
                   8: 'Priv-house-serv',
                   9: 'Prof-specialty',
                   10: 'Protective-serv',
                   11: 'Sales',
                   12: 'Tech-support',
                   13: 'Transport-moving'},
    'relationship': {0: 'Husband',
                     1: 'Not-in-family',
                     2: 'Other-relative',
                     3: 'Own-child',
                     4: 'Unmarried',
                     5: 'Wife'},
    'race': {0: 'Amer-Indian-Eskimo',
             1: 'Asian-Pac-Islander',
             2: 'Black',
             3: 'Other',
             4: 'White'},
    'sex': {0: 'Female', 1: 'Male'},
    'native-country': {0: 'Cambodia',
                       1: 'Canada',
                       2: 'China',
                       3: 'Columbia',
                       4: 'Cuba',
                       5: 'Dominican-Republic',
                       6: 'Ecuador',
                       7: 'El-Salvador',
                       8: 'England',
                       9: 'France',
                       10: 'Germany',
                       11: 'Greece',
                       12: 'Guatemala',
                       13: 'Haiti',
                       14: 'Holand-Netherlands',
                       15: 'Honduras',
                       16: 'Hong',
                       17: 'Hungary',
                       18: 'India',
                       19: 'Iran',
                       20: 'Ireland',
                       21: 'Italy',
                       22: 'Jamaica',
                       23: 'Japan',
                       24: 'Laos',
                       25: 'Mexico',
                       26: 'Nicaragua',
                       27: 'Outlying-US(Guam-USVI-etc)',
                       28: 'Peru',
                       29: 'Philippines',
                       30: 'Poland',
                       31: 'Portugal',
                       32: 'Puerto-Rico',
                       33: 'Scotland',
                       34: 'South',
                       35: 'Taiwan',
                       36: 'Thailand',
                       37: 'Trinadad&Tobago',
                       38: 'United-States',
                       39: 'Vietnam',
                       40: 'Yugoslavia'},
    'income': {0: '<=50K', 1: '>50K'}
}

map_s_i = {
    'workclass': {'Federal-gov': 0,
                  'Local-gov': 1,
                  'Private': 2,
                  'Self-emp-inc': 3,
                  'Self-emp-not-inc': 4,
                  'State-gov': 5,
                  'Without-pay': 6},
    'education': {'10th': 0,
                  '11th': 1,
                  '12th': 2,
                  '1st-4th': 3,
                  '5th-6th': 4,
                  '7th-8th': 5,
                  '9th': 6,
                  'Assoc-acdm': 7,
                  'Assoc-voc': 8,
                  'Bachelors': 9,
                  'Doctorate': 10,
                  'HS-grad': 11,
                  'Masters': 12,
                  'Preschool': 13,
                  'Prof-school': 14,
                  'Some-college': 15},
    'marital-status': {'Divorced': 0,
                       'Married-AF-spouse': 1,
                       'Married-civ-spouse': 2,
                       'Married-spouse-absent': 3,
                       'Never-married': 4,
                       'Separated': 5,
                       'Widowed': 6},
    'occupation': {'Adm-clerical': 0,
                   'Armed-Forces': 1,
                   'Craft-repair': 2,
                   'Exec-managerial': 3,
                   'Farming-fishing': 4,
                   'Handlers-cleaners': 5,
                   'Machine-op-inspct': 6,
                   'Other-service': 7,
                   'Priv-house-serv': 8,
                   'Prof-specialty': 9,
                   'Protective-serv': 10,
                   'Sales': 11,
                   'Tech-support': 12,
                   'Transport-moving': 13},
    'relationship': {'Husband': 0,
                     'Not-in-family': 1,
                     'Other-relative': 2,
                     'Own-child': 3,
                     'Unmarried': 4,
                     'Wife': 5},
    'race': {'Amer-Indian-Eskimo': 0,
             'Asian-Pac-Islander': 1,
             'Black': 2,
             'Other': 3,
             'White': 4},
    'sex': {'Female': 0, 'Male': 1},
    'native-country': {'Cambodia': 0,
                       'Canada': 1,
                       'China': 2,
                       'Columbia': 3,
                       'Cuba': 4,
                       'Dominican-Republic': 5,
                       'Ecuador': 6,
                       'El-Salvador': 7,
                       'England': 8,
                       'France': 9,
                       'Germany': 10,
                       'Greece': 11,
                       'Guatemala': 12,
                       'Haiti': 13,
                       'Holand-Netherlands': 14,
                       'Honduras': 15,
                       'Hong': 16,
                       'Hungary': 17,
                       'India': 18,
                       'Iran': 19,
                       'Ireland': 20,
                       'Italy': 21,
                       'Jamaica': 22,
                       'Japan': 23,
                       'Laos': 24,
                       'Mexico': 25,
                       'Nicaragua': 26,
                       'Outlying-US(Guam-USVI-etc)': 27,
                       'Peru': 28,
                       'Philippines': 29,
                       'Poland': 30,
                       'Portugal': 31,
                       'Puerto-Rico': 32,
                       'Scotland': 33,
                       'South': 34,
                       'Taiwan': 35,
                       'Thailand': 36,
                       'Trinadad&Tobago': 37,
                       'United-States': 38,
                       'Vietnam': 39,
                       'Yugoslavia': 40},
    'income': {'<=50K': 0, '>50K': 1}
}

census_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
]

dtype = [
    ('age', int),
    ('workclass', 'U20'),
    ('fnlwgt', int),
    ('education', 'U20'),
    ('education-num', int),
    ('marital-status', 'U20'),
    ('occupation', 'U20'),
    ('relationship', 'U20'),
    ('race', 'U20'),
    ('sex', 'U20'),
    ('capital-gain', int),
    ('capital-loss', int),
    ('hours-per-week', int),
    ('native-country', 'U20')
]