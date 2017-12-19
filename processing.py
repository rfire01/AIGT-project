import pandas as pd

from gains import vote_gain_all_candidates


RANK_DIST = {1: lambda df: df.VotesCand1PreVote,
             2: lambda df: df.VotesCand2PreVote,
             3: lambda df: df.VotesCand3PreVote}


def round_cmp_gain(df):
    """
        Calculate the gain for compromising on your second preference.

        Args:
            df(DataFrame): Round of one shot experiment.

        Return:
            float. The gain for compromising on your second preference.
    """
    ranks = [df.Pref1, df.Pref2, df.Pref3]
    utilities = [df.Util1, df.Util2, df.Util3]
    distribution = []

    for rank in ranks:
        distribution.append(RANK_DIST[rank](df))

    gains = vote_gain_all_candidates(distribution, utilities)
    return gains[1] - gains[0]


def find_scenario(df):
    """
        Return to which scenario a round belong to.

        Args:
            df(DataFrame): Round of one shot experiment.

        Return:
            float. The gain for compromising on your second preference.
    """
    pref1_poll = RANK_DIST[df.Pref1](df)
    pref2_poll = RANK_DIST[df.Pref2](df)
    pref3_poll = RANK_DIST[df.Pref3](df)

    # pref1 is the poll leader
    if pref1_poll >= pref2_poll and pref1_poll >= pref3_poll:
        if pref2_poll >= pref3_poll:
            return 'A'
        else:
            return 'B'
    # pref1 is last in poll
    elif pref1_poll < pref2_poll and pref1_poll < pref3_poll:
        if pref2_poll >= pref3_poll:
            return 'E'
        else:
            return 'F'
    # pref1 middle in poll
    else:
        if pref2_poll >= pref3_poll:
            return 'C'
        else:
            return 'D'


if __name__ == "__main__":
    data = pd.read_excel('OneShot\Filtered.xlsx')
    data['CmpGain'] = data.apply(round_cmp_gain, axis=1)
    data['Scenario'] = data.apply(find_scenario, axis=1)

    writer = pd.ExcelWriter("OneShot\Extended.xlsx")
    data.to_excel(writer, 'Sheet1', index=False)
    writer.save()
