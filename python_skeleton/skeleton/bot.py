'''
This file contains the base class that you should implement for your pokerbot.
'''

from .actions import *
from .states import *

import eval7 as val
import math
from eval7 import py_hand_vs_range_monte_carlo 

NUM_SIMULATIONS = 100_000
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
opp_bounty_prob = [1/13] * 13

class Bot():
    '''
    The base class for a pokerbot.
    '''

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        my_bounty = round_state.bounties[active]  # your current bounty rank

        ## reset opp_bounty_prob
        if round_num % 25 == 1:
            global opp_bounty_prob
            opp_bounty_prob = [1/13] * 13
            print(f"RESET BOUNTY PROBABILITIES: {opp_bounty_prob}")

        pass

    def update_bounty_prob(self, opp_hand, opp_bounty_hit):
        """
        Update the opponent's bounty probabilities based on the outcome of a hand.

        Parameters:
        - opp_hand (list of str): Observed card ranks in the current hand (e.g., ['2', '2', '4', '6']).
        - opp_bounty_hit (bool): True if the opponent's bounty was hit; False otherwise.
        - opp_bounty_prob (list of float): Current probabilities for each rank being the opponent's bounty.

        Returns:
        - updated_prob (list of float): Updated probabilities after considering the current hand.
        """
        
        # Create a mapping from rank to index for easy access
        rank_to_index = {rank: idx for idx, rank in enumerate(ranks)}
        
        # Initialize a dictionary to count occurrences of each rank in opp_hand
        rank_counts = {rank: 0 for rank in ranks}
        # print(f"Opponent hand: {opp_hand}")
        for card in opp_hand:
            rank = ranks[card.rank]
            if rank in rank_counts:
                rank_counts[rank] += 1
        # print (f"rank count: {rank_counts}")
        
        # Calculate the number of remaining cards in the deck
        remaining_deck_size = 52 - len(opp_hand)
        
        # Initialize a list to store unnormalized probabilities
        unnormalized_probs = []
        
        for rank in ranks:
            idx = rank_to_index[rank]
            count_seen = rank_counts[rank]
            remaining_count = 4 - count_seen  # Total of 4 cards per rank in a standard deck
            
            # If remaining_count is negative, it's an impossible scenario; set to 0
            if remaining_count < 0:
                remaining_count = 0
            
            if opp_bounty_hit:
                if count_seen > 0:
                    # If the rank is present in opp_hand, the bounty is definitely hit
                    likelihood = 1.0
                else:
                    # Probability that at least one card in opponent's hand is of this rank
                    # P(hit | rank) = 1 - P(no cards of rank in opponent's hand)
                    # P(no cards of rank) = C(remaining_deck_size - remaining_count, 2) / C(remaining_deck_size, 2)
                    try:
                        combinations_total = math.comb(remaining_deck_size, 2)
                        combinations_without_rank = math.comb(remaining_deck_size - remaining_count, 2)
                        prob_no_rank = combinations_without_rank / combinations_total
                        likelihood = 1.0 - prob_no_rank
                    except ValueError:
                        # Handle cases where remaining_deck_size < 2
                        likelihood = 0.0
            else:
                if count_seen > 0:
                    # If the rank is present in opp_hand, and bounty wasn't hit, this rank cannot be the bounty
                    likelihood = 0.0
                else:
                    # Probability that no cards in opponent's hand are of this rank
                    # P(not hit | rank) = C(remaining_deck_size - remaining_count, 2) / C(remaining_deck_size, 2)
                    try:
                        combinations_total = math.comb(remaining_deck_size, 2)
                        combinations_without_rank = math.comb(remaining_deck_size - remaining_count, 2)
                        likelihood = combinations_without_rank / combinations_total
                    except ValueError:
                        # Handle cases where remaining_deck_size < 2
                        likelihood = 1.0
            
            # Multiply by the prior probability
            prior_prob = opp_bounty_prob[idx]
            unnormalized_prob = likelihood * prior_prob
            unnormalized_probs.append(unnormalized_prob)
        
        # Calculate the total of unnormalized probabilities for normalization
        total_unnormalized = sum(unnormalized_probs)
        
        if total_unnormalized == 0:
            # Avoid division by zero; return the current probabilities unchanged
            # Alternatively, you could reset to uniform probabilities or handle as needed
            # print("Warning: Total unnormalized probability is zero. Returning prior probabilities unchanged.")
            return opp_bounty_prob.copy()
        
        # Normalize the probabilities so that they sum to 1
        updated_prob = [prob / total_unnormalized for prob in unnormalized_probs]

        for rank, prob in zip(ranks, updated_prob):
            print(f"Rank {rank}: {prob:.2%}")
        return updated_prob



    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        board_cards = previous_state.deck[:street]
        my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opp_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        opp_hand = board_cards+opp_cards
        OPP_HAND = [val.Card(s) for s in opp_hand]
        if my_delta < 0:
            global opp_bounty_prob
            opp_bounty_prob = self.update_bounty_prob(OPP_HAND, opp_bounty_hit)


    def generate_possible_hands(self, deck):
        ls = []
        for i in range(len(deck)):
            for j in range(len(deck)):
                if i == j:
                    continue
                temp = []
                temp.append(deck[i])
                temp.append(deck[j])
                ls.append([temp])
        return ls
    
    def compute_opp_bounty_hit_prob(self, opp_bounty_prob, my_cards, board_cards):

        rank_to_index = {rank: i for i, rank in enumerate(ranks)}  
        board_count = {r: 0 for r in ranks}
        my_count    = {r: 0 for r in ranks}
        
        for card in board_cards:
            # If card is '7c', card[0] = '7', card[1] = 'c'
            # If you store suits differently, adjust accordingly.
            r = card[0]  # rank character
            if r in board_count:
                board_count[r] += 1

        for card in my_cards:
            r = card[0]
            if r in my_count:
                my_count[r] += 1

        # 2. Number of known cards = my_cards + board_cards
        known_cards_count = len(my_cards) + len(board_cards)
        unknown_deck_size = 52 - known_cards_count

        opp_bounty_hit_prob = 0.0
        
        for r in ranks:
            idx = rank_to_index[r]
            # Probability that r is the opponent's bounty
            p_bounty = opp_bounty_prob[idx]
            if p_bounty <= 0.0:
                # If we've already deduced it's impossible or extremely unlikely,
                # no need to do more math
                continue
            
            # 3. Count how many copies of rank r are definitely seen in your hand + board
            total_seen = board_count[r] + my_count[r]
            
            # If the board already has at least 1 card of rank r,
            # then the opponent's final 5-card hand definitely "hits" that rank
            # (assuming we are at showdown or no more community cards to come).
            if board_count[r] > 0:
                p_hit_if_r = 1.0
            else:
                # Board does not show r.  The only way the opponent "hits" r
                # is if at least 1 of their 2 unknown hole cards is r.
                
                # 4. remaining_count = how many copies of rank r are left unseen in the deck
                remaining_count = 4 - total_seen
                if remaining_count <= 0:
                    # Means we've already accounted for all 4 copies in known cards,
                    # so there's no way the opponent has it
                    p_hit_if_r = 0.0
                else:
                    # Probability that the opponent has at least 1 copy of r 
                    # in their 2 unknown hole cards
                    #
                    #   = 1 - [C(unknown_deck_size - remaining_count, 2) 
                    #          / C(unknown_deck_size, 2)]
                    #
                    # (the complement of "none of the 2 hole cards are rank r")
                    if unknown_deck_size < 2:
                        # Edge case: if there's fewer than 2 unknown cards left in the deck
                        # (very unusual in normal workflow, but let's be safe)
                        p_hit_if_r = 0.0
                    else:
                        num_ways_without_r = math.comb(unknown_deck_size - remaining_count, 2)
                        num_ways_total     = math.comb(unknown_deck_size, 2)
                        p_no_r = num_ways_without_r / num_ways_total
                        p_hit_if_r = 1.0 - p_no_r
            
            # 5. Weighted by the probability that the bounty *is* r
            opp_bounty_hit_prob += p_bounty * p_hit_if_r

        return opp_bounty_hit_prob

    
    def calculate_pot_odds(self, opp_bounty_prob, equity, my_cards, board_cards, is_raise, 
                            my_contribution, opp_contribution, cost, bounty_hit):
        
        # Create a mapping from rank to index for easy access
        rank_to_index = {rank: idx for idx, rank in enumerate(ranks)}
        
        # Calculate P(bounty_hit): sum of opp_bounty_prob[r] for r in observed cards
        observed_ranks = set()
        # print(f"board cards: {board_cards}")
        for card in board_cards:
            observed_ranks.add(card[0])
        # print (f"observed ranks: {observed_ranks}")

        opp_bounty_hit = self.compute_opp_bounty_hit_prob(opp_bounty_prob, my_cards, board_cards)

        print(f"Probability opponent bounty is hit: {opp_bounty_hit:.3f}")
        
        # Compute expected_loss:
        # If opponent's bounty was hit and they win, their winnings are 1.5 * usual_winnings + 10
        # If bounty not hit and they win, their winnings are usual_winnings
        my_total = my_contribution+cost
        expected_loss = opp_bounty_hit * (1.5 * cost + 10) + (1.0 - opp_bounty_hit) * cost
        
        if not is_raise:
            if bounty_hit:
                ev = equity * (1.5 * opp_contribution + 10) - (1.0 - equity) * expected_loss
            else: 
                ev = equity * opp_contribution - (1.0 - equity) * expected_loss
        else: 
            if bounty_hit:
                ev = equity * (1.5 * my_total + 10) - (1.0 - equity) * expected_loss
            else: 
                ev = equity * my_total - (1.0 - equity) * expected_loss

        # Compute Pot Odds
        pot_odds = 1000  #check pot odds set to 1000 to always check if should and can
        if cost != 0:
            pot_odds = ev / cost
        return pot_odds


    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''

        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

        # newDeck = val.Deck()
        hand = [val.Card(s) for s in my_cards+board_cards]
        deck = val.Deck()
        for card in hand:
            deck.cards.remove(card)

        # Define the opponent's range as all possible two-card combinations remaining in the deck
        opponent_range = self.generate_possible_hands(deck)
        MY_CARDS = [val.Card(s) for s in my_cards]
        BOARD_CARDS = [val.Card(s) for s in board_cards]

        equity = py_hand_vs_range_monte_carlo(MY_CARDS, opponent_range, BOARD_CARDS, NUM_SIMULATIONS)
        print(f"Equity: {equity:.3f}")

        board_total = my_contribution+opp_contribution

        bounty_hit = False
        if my_bounty in hand:
            bounty_hit = True

        if CheckAction in legal_actions: 
            check_pot_odds = self.calculate_pot_odds(opp_bounty_prob, equity, my_cards, board_cards, False, 
                                            my_contribution, opp_contribution, continue_cost, bounty_hit)
        if CallAction in legal_actions:
            call_pot_odds = self.calculate_pot_odds(opp_bounty_prob, equity, my_cards, board_cards, False, 
                                            my_contribution, opp_contribution, continue_cost, bounty_hit)
            print(f"Call pot odds: {call_pot_odds:.3f}")
        if RaiseAction in legal_actions:
            all_in_pot_odds = self.calculate_pot_odds(opp_bounty_prob, equity, my_cards, board_cards, True, 
                                            my_contribution, opp_contribution, max_cost, bounty_hit)
            print(f"All in pot odds: {all_in_pot_odds:.3f}")
        
        
        # !!!! CHANGE ALL NUMBERS TO FLOAT. Ex: 2 --> 2.0

        # if not bounty_hit: 
        #     pot_odds = (continue_cost)/(board_total+continue_cost)
        #     if RaiseAction in legal_actions:
        #         min_raise_pot_odds = (min_cost)/(board_total+min_cost)
        #         all_in_pot_odds = (max_cost)/(board_total+max_cost)
        #         three_raise_pot_odds = 2
        #         ten_raise_pot_odds = 2
        #         thurty_raise_pot_odds = 2
        #         hundred_raise_pot_odds = 2
        #         if (3*min_cost < max_cost):
        #             three_raise_pot_odds = (3*min_cost)/(board_total+3*min_cost)
        #         if (10*min_cost < max_cost):
        #             ten_raise_pot_odds = (10*min_cost)/(board_total+10*min_cost)
        #         if (30*min_cost < max_cost):
        #             thirty_raise_pot_odds = (30*min_cost)/(board_total+30*min_cost)
        #         if (100*min_cost < max_cost):
        #            hundred_raise_pot_odds = (100*min_cost)/(board_total+100*min_cost) 
        # else: 
        #     pot_odds = (continue_cost)/(1.5*(board_total+continue_cost)+10)
        #     if RaiseAction in legal_actions:
        #         min_raise_pot_odds = (min_cost)/(1.5*(board_total+min_cost)+10)
        #         all_in_pot_odds = (max_cost)/(1.5*(board_total+max_cost)+10)
        #         three_raise_pot_odds = 2
        #         ten_raise_pot_odds = 2
        #         thurty_raise_pot_odds = 2
        #         hundred_raise_pot_odds = 2
        #         if (3*min_cost < max_cost):
        #             three_raise_pot_odds = (3*min_cost)/(1.5*(board_total+3*min_cost)+10)
        #         if (10*min_cost < max_cost):
        #             ten_raise_pot_odds = (10*min_cost)/(1.5*(board_total+10*min_cost)+10)
        #         if (30*min_cost < max_cost):
        #             thirty_raise_pot_odds = (30*min_cost)/(1.5*(board_total+30*min_cost)+10)
        #         if (100*min_cost < max_cost):
        #            hundred_raise_pot_odds = (100*min_cost)/(1.5*(board_total+100*min_cost)+10) 

        # somewhat gto bot

        # thresholds to include the fact that our opponent might have a good hand, variables subject to change
        all_in_threshold = 0.5 
        call_threshold = 0.1
        hundred_raise_threshold = 0.15
        thirty_raise_threshold = 0.12
        ten_raise_threshold = 0.09
        three_raise_threshold = 0.06
        min_raise_threshold = 0.04

        # # raise equity
        # if RaiseAction in legal_actions:
        #     if equity > all_in_pot_odds+all_in_threshold:
        #         return RaiseAction(max_raise)
        #     if equity > hundred_raise_pot_odds+hundred_raise_threshold:
        #         return RaiseAction(100*min_cost)
        #     if equity > thirty_raise_pot_odds+thirty_raise_threshold:
        #         return RaiseAction(30*min_cost)
        #     if equity > ten_raise_pot_odds+ten_raise_threshold:
        #         return RaiseAction(10*min_cost)
        #     if equity > three_raise_pot_odds+three_raise_threshold:
        #         return RaiseAction(3*min_cost)
        #     if equity > min_raise_pot_odds+min_raise_threshold:
        #         return RaiseAction(min_raise)
            
        # # call equity
        # if (equity > pot_odds+call_threshold and CallAction in legal_actions):
        #     return CallAction()
        
        # # check/fold
        # if CheckAction in legal_actions: 
        #     return CheckAction()
        # else: 
        #     return FoldAction()

        
        # ALL IN BOT
        if RaiseAction in legal_actions:
            if all_in_pot_odds >= all_in_threshold:
                return RaiseAction(max_raise)
        if CallAction in legal_actions: 
            if call_pot_odds >= call_threshold:
                return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        else: 
            return FoldAction()

        # # Random bot
        # if RaiseAction in legal_actions:
        #     if random.random() < 0.5:
        #         return RaiseAction(min_raise)
        # if CheckAction in legal_actions:  # check-call
        #     return CheckAction()
        # if random.random() < 0.25:
        #     return FoldAction()
        # return CallAction()