import os
from itertools import izip

from sklearn import svm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_features(user, wanted_scenario='F'):
    scenarios, actions, gains, votes1, votes2, total_votes = user
    result_arr = []
    for index, vote_result in enumerate(izip(scenarios, actions, gains)):
        s, a, _ = vote_result
        if s == wanted_scenario:
            mean_action_value = np.mean(map(
                lambda x: (3 - x),
                actions[:index] + actions[index + 1:]))

            result_arr.append([mean_action_value,
                               np.mean(gains[:index] + gains[index + 1:]),
                               np.mean([((v1 + v2) / t_v) for v1, v2, t_v in
                                        izip(votes1, votes2, total_votes)]),
                               float(votes1[index]) / total_votes[index],

                               a]) # <- result

    return result_arr

def filter_users(user):
    scenarios, actions, gains, votes1, votes2, total_votes = user
    m_a = -1
    for s,a in izip(scenarios, actions):
        if s == 'F':
            if m_a == -1:
                m_a = a
            elif m_a != a:
                return False
    return True

if __name__ == "__main__":
    data_dir = os.path.join(os.path.abspath('.'), 'OneShot')
    data_path = os.path.join(data_dir, 'PreML.xlsx')
    df = pd.read_excel(data_path)

    grouped = df.groupby(['VoterID'])
    user_arr = []
    for ID, voter in grouped:
        user_arr.append(([s for s in voter['Scenario']],
                         [ac for ac in voter['Action']],
                         [g for g in voter['CmpGain']],
                         [v1 for v1 in voter['VotesCand1PreVote']],
                         [v2 for v2 in voter['VotesCand2PreVote']],
                         [v2 for v2 in voter['NumVotes']]))

    user_arr = filter(filter_users, user_arr)
    print 'all_size: {}'.format(len(user_arr))
    for _ in xrange(10):
        train, test = train_test_split(user_arr, test_size=0.2)
        train_set = reduce(lambda x, y: x + y, map(create_features, train))
        test_set = reduce(lambda x, y: x + y, map(create_features, test))
        feature_number = len(train_set[0]) - 1

        svc = svm.SVC(kernel='linear')
        svc.fit([x[:feature_number] for x in train_set], [x[feature_number] for x in train_set])

        result = svc.predict([x[:feature_number] for x in test_set])

        print 'rate: {}'.format(float(
            sum([1 for x, y in izip(test_set, result) if x[feature_number] == y])) / len(result))
