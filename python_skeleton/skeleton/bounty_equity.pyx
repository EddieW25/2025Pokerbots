# bounty_equity.pyx
# (You will compile this with setup.py or a similar mechanism.)

# We assume Python 3 semantics:
#cython: language_level=3

from random import random
import eval7 as val
# If "Card" is a cdef class or python class from val, you might need:
# cimport val  # if val has cdef classes you want to access at the C level

#--------------------------------------------------------------------
# 1) Weighted pick of a two-card hand from opponent_range
#--------------------------------------------------------------------
cdef list pick_weighted_two_card_hand(dict opponent_range):
    """
    opponent_range is a dict where:
        key   = (Card, Card)
        value = float (the weight)
    This function returns a list of two Card objects [Card, Card],
    chosen randomly according to the distribution of weights.
    """

    cdef double total_weight = 0.0
    cdef double cumulative = 0.0
    cdef double r
    cdef object key   # the key in the dict
    cdef double w     # the weight in the dict

    # 1) Sum up all weights
    for key, w in opponent_range.items():
        total_weight += w

    # Edge case: if total_weight is 0 or negative, just fallback
    # to any random key. Usually means an empty or invalid range.
    if total_weight <= 0.0:
        # Return an empty list or raise an exception
        return []

    # 2) Generate a random float in [0, total_weight)
    r = random() * total_weight

    # 3) Walk through opponent_range until we cross r
    cumulative = 0.0
    for key, w in opponent_range.items():
        cumulative += w
        if r <= cumulative:
            # 'key' should be a tuple of two Cards: (Card, Card)
            # We'll return them as a Python list
            return [key[0], key[1]]

    # In case of any floating rounding issues, fallback to the last one
    return [key[0], key[1]]


#--------------------------------------------------------------------
# 2) Main Monte Carlo equity function
#--------------------------------------------------------------------
cpdef tuple get_equity_and_opp_bounty_prob(
        list my_cards, 
        dict opponent_range,      # e.g. {(Card('2c'), Card('2h')): 0.0013, ...}
        list board_cards,         # e.g. [Card('8d'), Card('Js')]
        list opp_bounty_prob,     # e.g. 13 floats: probability of each rank
        int NUM_SIMULATIONS
    ):
    """
    Returns (equity, opp_bounty_hit) after a Monte Carlo simulation.

    * equity = (wins + 0.5 * ties) / NUM_SIMULATIONS
    * opp_bounty_hit = average probability that the opponent's bounty rank
                       appears in opp_hand + final board

    Steps:
      1) For each simulation:
         (a) Pick a 2-card hand for the opponent from the weighted distribution
         (b) Create a shuffled deck, pick needed_board_cards from the top
         (c) Combine to get my 7 cards vs. opp's 7 cards
         (d) Evaluate each
         (e) Tally wins/ties for hero
         (f) Compute partial 'opp_bounty_hit_prob' for this sim
      2) Return final average
    """
    cdef int wins = 0
    cdef int ties = 0
    cdef double opp_bounty_hit_total = 0.0
    cdef int needed_board_cards = 5 - len(board_cards)
    cdef int i, j
    cdef double opp_bounty_hit_prob
    cdef list opp_cards        # We'll store the 2-card pick
    cdef list all_cards
    cdef list board
    cdef list my_hand
    cdef list opp_hand
    cdef int my_eval
    cdef int opp_eval
    cdef int rank_i
    cdef int opp_rank_counts[13]  # track which ranks appear in opp_hand

    for i in range(NUM_SIMULATIONS):
        # 1) Pick villain's 2-card combo from distribution
        opp_cards = pick_weighted_two_card_hand(opponent_range)
        if not opp_cards:
            # If empty => no valid pick, skip or break
            continue

        # 2) Create deck & shuffle
        deck = val.Deck()
        deck.shuffle()

        # 3) Build final board. 
        #    The original code snippet does NOT remove known cards from the deck;
        #    We replicate that logic literally here.
        board = board_cards.copy()
        for j in range(needed_board_cards):
            board.append(deck.cards[j])

        # 4) Combine for each player's final 7 cards
        my_hand = my_cards + board
        opp_hand = opp_cards + board

        # 5) Calculate probability of hitting the bounty rank
        opp_bounty_hit_prob = 0.0
        for j in range(13):
            opp_rank_counts[j] = 0

        # Go through each card in opp_hand
        # We assume card.rank is an int in 0..12 or 1..13
        # If it's 2..14, adapt as needed (e.g. "r = card.rank - 2").
        for c in opp_hand:
            rank_i = c.rank   # c is a Card object
            # if ranks are 2..14, do: rank_i = c.rank - 2
            if opp_rank_counts[rank_i] == 0:
                opp_rank_counts[rank_i] = 1
                opp_bounty_hit_prob += opp_bounty_prob[rank_i]

        opp_bounty_hit_total += opp_bounty_hit_prob

        # 6) Evaluate each hand (7 cards). 
        #    We assume val.evaluate(...) returns an integer strength
        my_eval = val.evaluate(my_hand)
        opp_eval = val.evaluate(opp_hand)

        if my_eval > opp_eval:
            wins += 1
        elif my_eval == opp_eval:
            ties += 1

    # Compute final metrics
    cdef double equity = 0.0
    cdef double opp_bounty_hit = 0.0
    if NUM_SIMULATIONS > 0:
        equity = (wins + 0.5 * ties) / NUM_SIMULATIONS
        opp_bounty_hit = opp_bounty_hit_total / NUM_SIMULATIONS

    return (equity, opp_bounty_hit)