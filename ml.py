import os
import numpy as np
import pandas as pd
from sklearn import svm
from itertools import izip
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

RANK_DIST = {1: lambda df: df.VotesCand1PreVote,
             2: lambda df: df.VotesCand2PreVote,
             3: lambda df: df.VotesCand3PreVote}


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
                               float(votes1[index] - votes2[index]) / total_votes[index],


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
                         [v1 for v1 in voter.apply(lambda row: RANK_DIST[row['Pref1']](row), axis=1)],
                         [v2 for v2 in voter.apply(lambda row: RANK_DIST[row['Pref2']](row), axis=1)],
                         [v2 for v2 in voter['NumVotes']]))

    features = reduce(lambda x, y: x + y, map(create_features, user_arr))
    inputs = np.array([x[:-1] for x in features])
    outputs = np.array([x[-1] for x in features])
    kf = KFold(n_splits=10)


    cm =[]
    for train_index, test_index in kf.split(inputs):
        in_train, in_test = inputs[train_index], inputs[test_index]
        out_train, out_test = outputs[train_index], outputs[test_index]

        svc = svm.SVC(kernel='linear')
        svc.fit(in_train, out_train)

        prediction = svc.predict(in_test)
        cm.append(confusion_matrix(out_test, prediction, labels=[1, 2, 3]))

    final_cm = sum(cm)
    print sum(cm)
    tp = final_cm[0][0]
    fn = final_cm[0][1] + final_cm[0][2]
    fp = final_cm[1][0] + final_cm[2][0]

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    fmeasure = 2 * precision * recall / (precision + recall)

    print precision
    print recall
    print fmeasure
