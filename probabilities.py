import math
from scipy.stats import multinomial


def tie_probability(distribution, candidate1, candidate2):
    """
    Calculate the probability for tie between 2 candidates.

    Args:
        distribution(list): Distribution over three candidates.
        candidate1(int): Index of the first candidate for the tie.
        candidate2(int): Index of the second candidate for the tie.

   Return:
       double. the probability that both candidate1 and candidate2 win.
    """
    candidates_index = [candidate1, candidate2]
    candidate3 = [x for x in range(len(distribution)) if
                  x not in candidates_index][0]

    total_amount = float(sum(distribution))
    candidates_probability = [distribution[cand] / total_amount for cand in
                              [candidate1, candidate2, candidate3]]

    start_value = int(math.ceil(total_amount / 3.0))
    tie_probability_value = 0
    for votes in range(start_value, int(total_amount) / 2 + 1):
        state = [votes, votes, total_amount - 2 * votes]
        state_probability = multinomial.pmf(state, total_amount,
                                            candidates_probability)
        tie_probability_value += state_probability

    return tie_probability_value


def close_tie_probability(distribution, candidate1, candidate2):
    """
    Calculate the probability that candidate2 have one more vote than
        candidate1.

    Args:
        distribution(list): Distribution over three candidates.
        candidate1(int): Index of the first candidate for the tie.
        candidate2(int): Index of the second candidate for the tie.

   Return:
       double. the probability that candidate2 wins and candidate1 have one
            less vote.
    """

    candidates_index = [candidate1, candidate2]
    candidate3 = [x for x in range(len(distribution)) if
                  x not in candidates_index][0]

    total_amount = float(sum(distribution))
    candidates_probability = [distribution[cand] / total_amount for cand in
                              [candidate1, candidate2, candidate3]]

    start_value = int(math.ceil((total_amount + 1) / 3.0)) + 1
    close_tie_probability_value = 0
    for votes in range(start_value, int(total_amount + 1) / 2 + 1):
        state = [votes - 1, votes, total_amount - 2 * votes + 1]
        state_probability = multinomial.pmf(state, total_amount,
                                            candidates_probability)
        close_tie_probability_value += state_probability

    return close_tie_probability_value
