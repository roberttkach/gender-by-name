import pandas as pd
import numpy as np


def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)

    names = df.iloc[:, 0].values
    genders = df.iloc[:, 1].values

    indices = [i for i, name in enumerate(names) if len(name) > 12]
    names = np.delete(names, indices)
    genders = np.delete(genders, indices)

    longest_word = max(names, key=len)

    women = []
    men = []
    for i in range(len(genders)):
        if genders[i] == 'F':
            women.append([names[i], genders[i]])
        else:
            men.append([names[i], genders[i]])

    size = min(len(men), len(women))
    var = men[:size]
    var = women[:size]

    genders_bin = [1 if gender == 'M' else 0 for gender in genders]

    return names, genders_bin, longest_word
