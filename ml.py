import os
from itertools import izip

from sklearn import svm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_features(user, wanted_scenario='F'):
    scenarios, actions, gains = user
    result_arr = []
    for index, vote_result in enumerate(izip(scenarios, actions, gains)):
        s, a, _ = vote_result
        if s == wanted_scenario:
            mean_action_value = np.mean(map(
                lambda x: (3 - df.Action) * 10,
                actions[:index] + actions[index + 1:])) / 20

            result_arr.append([mean_action_value,
                               np.mean(gains[:index] + gains[index + 1:]),
                               a])

    return result_arr


if __name__ == "__main__":
    data_dir = os.path.join(os.path.abspath('.'), 'OneShot')
    data_path = os.path.join(data_dir, 'PreML.xlsx')
    df = pd.read_excel(data_path)

    grouped = df.groupby(['VoterID'])
    user_arr = []
    for ID, voter in grouped:
        user_arr.append(([s for s in voter['Scenario']],
                         [ac for ac in voter['Action']],
                         [g for g in voter['CmpGain']]))

    for _ in xrange(10):
        train, test = train_test_split(user_arr, test_size=0.2)

        train_set = reduce(lambda x, y: x + y, map(create_features, train))
        test_set = reduce(lambda x, y: x + y, map(create_features, test))

        svc = svm.SVC(kernel='linear')
        svc.fit([x[:2] for x in train_set], [x[2] for x in train_set])

        result = svc.predict([x[:2] for x in test_set])

        print 'rate: {}'.format(float(
            sum([1 for x, y in izip(test_set, result) if x[2] == y])) / len(result))
