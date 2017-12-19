from probabilities import tie_probability as tp, close_tie_probability as ctp


def calculate_gain(winners_utilities):
    """
        Calculate the gain for the candidates in winners.

        Args:
            winners_utilities(list) - Utility for each candidate that won.

        Returns:
            float. Gain for the winning candidates.
    """
    return float(sum(winners_utilities)) / len(winners_utilities)


def vote_gain_single_candidate(distribution, utilities, candidate_index):
    """
        Calculate the gain for voting candidate in candidate_rank rank.

        Args:
            distribution(list) - Distribution over candidate.
            utilities(list) - Utility for each candidate in distribution.
            candidate_index(int) - Candidate index in distribution.

        Return:
            float. The gain for voting to the candidate that ranked as
                candidate_rank.
    """
    opponents = [x for x in range(len(distribution)) if x != candidate_index]

    gain = 0
    for opponent in opponents:
        tie_prob = tp(distribution, candidate_index, opponent)
        close_tie_prob = ctp(distribution, candidate_index, opponent)

        chosen_utility = calculate_gain([utilities[candidate_index]])
        tie_utility = calculate_gain([utilities[candidate_index],
                                      utilities[opponent]])
        opponent_utility = calculate_gain([utilities[opponent]])

        gain = gain + tie_prob * (chosen_utility - tie_utility)
        gain = gain + close_tie_prob * (tie_utility - opponent_utility)

    return gain


def vote_gain_all_candidates(distribution, utilities):
    """
        Calculate the gain of voting to each candidate.

        Args:
            distribution(list) - Distribution over candidate.
            utilities(list) - Utility for each candidate in ranks.

        Return:
            list. The gain for voting to each candidate.
    """
    gains = [vote_gain_single_candidate(distribution, utilities, candidate) for
             candidate in range(len(utilities))]
    return gains
