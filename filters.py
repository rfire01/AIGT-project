import pandas as pd


def filter_under_participants(df, n):
    """
        Filter From DataFrame all rounds with more then n participants.

        Args:
             df(DataFrame): Holds the data from all rounds.
             n(int): Maximum amount of participants.

        Return:
             DataFrame. Filtered data.
    """
    return df[df.NumVotes < n]


def filter_atleast_rounds(df, r):
    """
        Filter From DataFrame all users that did less than r rounds.

        Args:
             df(DataFrame): Holds the data from all rounds.
             r(int): Minimum amount of rounds for each user.

        Return:
             DataFrame. Filtered data.
    """
    tmp_df = df.groupby(["VoterID"]).count()
    for row in tmp_df.itertuples():
        if row[1] < r:
            df = df[df.VoterID != row[0]]

    return df


def run_filters(files, output_path):
    """
        Run all filtes on the combination of all files.

        Args:
             files(list): All excel files need to be filtered.
             output_path(str): Output path.

        Return:
             DataFrame. Filtered data.
    """
    df = pd.DataFrame()
    for excel in files:
        df = df.append(pd.read_excel(excel))

    df = filter_under_participants(df, 10000)
    df = filter_atleast_rounds(df, 15)

    writer = pd.ExcelWriter(output_path)
    df.to_excel(writer, 'Sheet1', index=False)
    writer.save()


if __name__ == "__main__":
    run_filters(["OneShot\Book1.xlsx"], "OneShot\Filtered.xlsx")