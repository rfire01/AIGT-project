import os

import pandas as pd


def remove_without_scenario(df, scenario='F'):
    """
        Filter From DataFrame all voters that vote didn't encounter
            the scenario.

        Args:
             df(DataFrame): Holds the data from all rounds.
             scenario (str): the scenario to test

        Return:
             DataFrame. Filtered data.
    """
    grouped = df.groupby(['VoterID'])
    for ID, voter in grouped:
        if scenario not in [s for s in voter['Scenario']]:
            df = df[df.VoterID != ID]

    return df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.abspath('.'), 'OneShot')
    data_path = os.path.join(data_dir, 'Extended.xlsx')
    output_file = os.path.join(data_dir, 'PreML.xlsx')
    data = pd.read_excel(data_path)
    data = remove_without_scenario(data)
    writer = pd.ExcelWriter(output_file)
    data.to_excel(writer, 'Sheet1', index=False)
    writer.save()

