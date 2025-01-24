# bounty_equity.pyx
# Compile with:
#   python setup.py build_ext --inplace
# or something similar.

import cython
import eval7
import math
from random import random

from eval7.cards cimport cards_to_mask
from eval7.evaluate cimport cy_evaluate
from eval7.xorshift_rand cimport randint

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void *malloc(size_t n_bytes)
    void free(void *ptr)

# A small library array for card bitmasks
cdef unsigned long long card_masks_table[52]

cdef unsigned int load_card_masks():
    cdef int i
    for i in range(52):
        card_masks_table[i] = 1 << i
    return 0

load_card_masks()

@cython.cfunc
def c_weight_hand(unsigned int rank1, unsigned int suit1,
                  unsigned int rank2, unsigned int suit2,
                  list board_cards) -> double:
    """
    This is a cfunc to replicate your weighting logic. 
    We'll produce a single float (double) weight for the 2-card hand
    given the board. 
    We assume rank1..rank2 in [0..12], suit1..suit2 in [0..3].
    'board_cards' is a Python list of Card objects, each with .rank, .suit.
    """

    cdef double w = 1.0
    cdef unsigned int i
    cdef unsigned int r1 = rank1
    cdef unsigned int r2 = rank2
    cdef unsigned int s1 = suit1
    cdef unsigned int s2 = suit2

    # replicate your rank_count, suit_count logic
    cdef int rank_count[13]
    cdef int suit_count[4]

    for i in range(13):
        rank_count[i] = 0
    for i in range(4):
        suit_count[i] = 0

    # fill from board
    cdef int n = len(board_cards)
    cdef unsigned int brank, bsuit
    for i in range(n):
        brank = board_cards[i].rank
        bsuit = board_cards[i].suit
        rank_count[brank] += 1
        suit_count[bsuit] += 1

    # fill from the 2-card combo
    rank_count[r1] += 1
    rank_count[r2] += 1
    suit_count[s1] += 1
    suit_count[s2] += 1

    # replicate your logic:

    # pockets
    cdef bint pocket = False
    if r1 == r2:
        w = 1.4 + r1/10.0
        pocket = True

    # quads
    if rank_count[r1] == 4 or rank_count[r2] == 4:
        w = 25.0

    cdef bint trips = False
    cdef int tripIndex = -1
    if (rank_count[r1] == 3 or rank_count[r2] == 3):
        w = 4.5 if pocket else 4.0
        trips = True
        tripIndex = r1 if rank_count[r1] == 3 else r2

    # board-based pair/trips logic
    cdef int j
    for j in range(n):
        brank = board_cards[j].rank
        # full house check
        if trips and rank_count[brank] >= 2 and brank != tripIndex:
            w = 15.0
            break
        # pairs
        if r1 == brank:
            w *= (1.2 + r1/20.0)
        if r2 == brank:
            w *= (1.2 + r2/20.0)

    cdef bint flush = False
    # flush logic
    for j in range(4):
        if suit_count[j] == 4:
            w *= 3.5 if n == 3 else 2.4
        elif suit_count[j] == 5:
            w = 10.0
            flush = True

    # straight draw logic
    cdef int psums[14]
    psums[0] = 0
    for j in range(13):
        psums[j+1] = psums[j] + min(1, rank_count[j])

    for j in range(5,14):
        cdef int val = psums[j] - psums[j-5]
        if val == 4:
            w *= 1.6 if n == 3 else 1.3
        elif val == 5:
            w = 8.0 if not flush else 35.0

    return w


@cython.cfunc
def c_generate_weighted_dict(list deck,
                             list my_hand,
                             list board_cards) -> dict:
    """
    Generates all possible combos from 'deck' excluding 'my_hand',
    then weighs them using c_weight_hand(...) with board_cards.
    Returns a dict { (Card,Card): weight } in Python.

    In real usage, you'd want more efficient combos, 
    but this mirrors your original approach of nested loops 
    minus the overhead of a large Pythonic approach.
    """
    cdef int i, j
    cdef int nd = len(deck)
    cdef dict result = {}
    cdef object c1, c2
    for i in range(nd):
        c1 = deck[i]
        for j in range(i+1, nd):
            c2 = deck[j]
            # ensure c1,c2 not in my_hand
            if c1 not in my_hand and c2 not in my_hand:
                # compute weight
                cdef double w = c_weight_hand(c1.rank, c1.suit,
                                              c2.rank, c2.suit,
                                              board_cards)
                # store
                result[(c1, c2)] = w

    return result


@cython.ccall
def estimate_opponent_range(
    list my_hand,
    list board_cards,
    str action
):
    """
    Example ccall function that:
    1) builds a deck
    2) generates combos
    3) weights them
    4) prunes them based on action
    5) returns a python dict of { (Card,Card): weight }.
    """
    cdef list deck = eval7.Deck().cards  # or your own method
    cdef dict raw_dict = c_generate_weighted_dict(deck, my_hand, board_cards)

    # update_range_based_on_action logic:
    cdef dict final_dict = {}
    cdef double min_threshold = 1.0
    if action == 'raise':
        min_threshold = 2.0
    elif action == 'call':
        min_threshold = 1.5
    # else fold => no combos

    cdef object combo
    for combo, w in raw_dict.items():
        if w >= min_threshold:
            final_dict[combo] = w

    return final_dict


def py_estimate_opponent_range(my_hand, board_cards, action):
    """
    Python-callable wrapper
    """
    return estimate_opponent_range(my_hand, board_cards, action)