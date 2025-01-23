import eval7
import cython
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

cdef unsigned int filter_options(unsigned long long *source,
        unsigned long long *target,
        unsigned int num_options,
        unsigned long long dead):
    """
    Removes all options that share a dead card
    Returns total number of options kept
    """
    cdef unsigned long long option
    cdef unsigned int total = 0
    cdef unsigned int s
    for s in range(num_options):
        option = source[s]
        # if it doesn't overlap with 'dead', we keep it
        if (option & dead) == 0:
            target[total] = option
            total += 1
    return total


cdef unsigned long long deal_card(unsigned long long dead):
    cdef unsigned int cardex
    cdef unsigned long long card
    while True:
        cardex = randint(52)   # 0..51
        card = card_masks_table[cardex]
        if (dead & card) == 0:
            return card


cdef void build_weighted_arrays(dict range_dict,
                                unsigned long long** combos_out,
                                double** prefix_out,
                                int* n_out,
                                double* total_out):
    """
    Take a Python dict { (Card,Card)->weight } or {some 2-card rep -> weight}.
    For each entry:
      - convert the 2-card combo to a mask
      - accumulate in a c-array
      - build prefix sums in another c-array (both length n).
    Output:
      combos_out -> c array of 'unsigned long long' for each combo
      prefix_out -> c array of double prefix sums
      n_out -> number of combos
      total_out -> final total weight
    """
    cdef int n = len(range_dict)
    cdef unsigned long long* combos = <unsigned long long*> malloc(n * sizeof(unsigned long long))
    cdef double* prefix = <double*> malloc(n * sizeof(double))

    cdef double running = 0.0
    cdef int i = 0

    cdef unsigned long long mask
    cdef double w

    # We'll iterate over the dictionary items
    for combo, weight in range_dict.items():
        mask = 0
        # e.g. combo[0] is the first Card
        mask |= cards_to_mask(combo[0:1])  # mask for single card
        mask |= cards_to_mask(combo[1:2])  # combine

        w = weight
        running += w
        prefix[i] = running
        combos[i] = mask
        i += 1

    combos_out[0] = combos
    prefix_out[0] = prefix
    n_out[0] = n
    total_out[0] = running


from libc.math cimport floor

cdef inline unsigned long long pick_weighted_combo(unsigned long long* combos,
                                                   double* prefix,
                                                   int n,
                                                   double total):
    """
    Sample a single 2-card combo from combos[] with distribution
    given by prefix[] (prefix sums). 'total' is prefix[n-1].
    We'll do a random draw in [0, total), do a manual binary search.
    Return the chosen combo mask.
    """
    cdef double r = (randint(10000000) / 10000000.0) * total
    # or you could do a real floating random if you had it:
    #   r = random() * total

    cdef int left = 0
    cdef int right = n - 1
    cdef int mid
    cdef double val
    while left < right:
        mid = (left + right) >> 1
        val = prefix[mid]
        if r <= val:
            right = mid
        else:
            left = mid + 1
    return combos[left]

cdef double opp_bounty_hit = 0.0

cdef double hand_vs_weighted_range_monte_carlo(unsigned long long hero,
                                              unsigned long long* combos,
                                              double* prefix,
                                              int n,
                                              double total_weight,
                                              unsigned long long start_board,
                                              int num_board,
                                              double* array_ptr,
                                              int iterations):
    """
    Monte Carlo for a WEIGHTED range.
    We'll sample from combos[] prefix-sums each iteration, then deal out the board.
    """
    cdef unsigned int count = 0
    cdef double opp_bounty_hit_total = 0.0
    cdef unsigned int i, j

    cdef unsigned long long villain_combo
    cdef unsigned long long dealt
    cdef unsigned long long board
    cdef unsigned int hero_val
    cdef unsigned int vill_val

    cdef int needed = 5 - num_board
    if needed < 0:
        needed = 0

    cdef unsigned long long villain_final  # bitmask for villain's 7 cards

    # We'll parse villain_final to see which ranks appear
    cdef int rank_used[13]
    cdef double opp_bounty_hit_prob
    cdef int card_idx, rank_idx

    for i in range(iterations):
        # 1) pick random villain combo from weighted prefix
        villain_combo = pick_weighted_combo(combos, prefix, n, total_weight)

        # 2) deal the rest of the board
        dealt = hero | villain_combo
        board = start_board
        for j in range(needed):
            board |= deal_card(board | dealt)

        # 3) Evaluate
        hero_val = cy_evaluate(board | hero, 7)
        vill_val = cy_evaluate(board | villain_combo, 7)

        if hero_val > vill_val:
            count += 2
        elif hero_val == vill_val:
            count += 1
        
        # 4) Calulate opp_bounty_hit
        villain_final = villain_combo | board
        for j in range(13):
            rank_used[j] = 0

        opp_bounty_hit_prob = 0.0

        # For each card index 0..51, check if that bit is set
        # If set, find rank => card_idx // 4 => 0..12
        # If rank not used yet, add array_ptr[rank_idx]
        for card_idx in range(52):
            if (villain_final & (1 << card_idx)) != 0:
                rank_idx = card_idx // 4
                if rank_used[rank_idx] == 0:
                    rank_used[rank_idx] = 1
                    opp_bounty_hit_prob += array_ptr[rank_idx]

        opp_bounty_hit_total += opp_bounty_hit_prob
    
    cdef double eq = 0.5 * count / iterations
    # average bounty hit probability
    opp_bounty_hit = opp_bounty_hit_total / iterations

    return eq


def py_hand_vs_weighted_range_monte_carlo(py_hand, 
                                          py_villain_dict,  # dict of {(Card,Card) : weight}
                                          py_board,
                                          opp_bounty_prob, 
                                          py_iterations):
    """
    Python entry for Weighted Range Monte Carlo.
    1) Build c-arrays of combos + prefix sums.
    2) filter out combos that conflict with hero or start_board
    3) run hand_vs_weighted_range_monte_carlo
    4) free memory
    """
    cdef unsigned long long hero_mask = cards_to_mask(py_hand)
    cdef unsigned long long board_mask = cards_to_mask(py_board)
    cdef int num_board = len(py_board)
    cdef int iterations = py_iterations

    cdef unsigned long long* combos = NULL
    cdef double* prefix = NULL
    cdef int n = 0
    cdef double total_weight = 0.0

    # (A) Build arrays
    build_weighted_arrays(py_villain_dict, &combos, &prefix, &n, &total_weight)
    if n == 0 or total_weight <= 0:
        # no valid combos, trivially
        return 1.0  # or 0.0, your choice

    # (B) filter out combos that overlap with hero or known board
    cdef unsigned long long dead = hero_mask | board_mask

    # We'll do in-place filter in combos + prefix:
    # We'll just move combos/prefix that are valid to the front, track 'cnt'
    cdef int cnt = 0
    cdef int idx
    cdef unsigned long long ccombo
    cdef double run = 0.0
    cdef double prev = 0.0
    for idx in range(n):
        ccombo = combos[idx]
        if (ccombo & dead) == 0:  # no overlap
            # keep it
            combos[cnt] = ccombo
            run = prefix[idx]  # the cumulative sum at idx
            # We want the "increment" from the previous. If idx==0 => prefix=some val.
            # But simpler is to do prefix of the new array from scratch:
            if cnt == 0:
                prefix[cnt] = run - 0.0
            else:
                prefix[cnt] = prefix[cnt - 1] + (run - prev)
            prev = run
            cnt += 1

    n = cnt
    if n == 0:
        free(combos)
        free(prefix)
        return 1.0  # or 0.0

    total_weight = prefix[n - 1]  # final prefix sum

    # (C) run the weighted Monte Carlo
    cdef:
        Py_ssize_t length = len(opp_bounty_prob)  # get size from python container
        double* array_ptr
        int i
    array_ptr = <double*> malloc(length * cython.sizeof(double)) 
    # manually copy
    for i in range(length):
        array_ptr[i] = <double> opp_bounty_prob[i]
    free(array_ptr)
    cdef double eq
    eq= hand_vs_weighted_range_monte_carlo(hero_mask,
                                                       combos,
                                                       prefix,
                                                       n,
                                                       total_weight,
                                                       board_mask,
                                                       num_board,
                                                       array_ptr,
                                                       iterations)
    # (D) free
    free(combos)
    free(prefix)

    return eq, opp_bounty_hit