'''
This file contains the base class that you should implement for your pokerbot.
'''

from .actions import *
from .states import *

import eval7 as val
import math
import random 
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "python_skeleton", "skeleton"))
from .bounty_equity import py_hand_vs_weighted_range_monte_carlo 
from itertools import combinations
# from eval7 import py_hand_vs_range_monte_carlo 
import bisect

NUM_SIMULATIONS = 100_000
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suits = ['s', 'h', 'd', 'c']
opp_bounty_prob = np.array([1/13]*13, dtype=np.float64)
past_bankroll = 0
weights_list = []

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
        rounds_left = 1001-round_num
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        my_bounty = round_state.bounties[active]  # your current bounty rank
        
        print(f"ROUND NUM: {round_num}")
        print(f"TIME LEFT: {game_clock}")

        # reset opp_bounty_prob
        if round_num % 25 == 1:
            global opp_bounty_prob
            opp_bounty_prob = np.array([1/13]*13, dtype=np.float64)
            print(f"RESET BOUNTY PROBABILITIES: {opp_bounty_prob}")
        
        # # reset past_bankroll every 200 rounds, and change strategy if needed
        # global past_bankroll
        # if round_num % 200 == 1:
        #     ## change strategy code here
        #     past_bankroll = 0


        # # if we can always fold and guarantee a win
        # if (rounds_left < 30 and my_bankroll > 75):
        #     # always fold
        # if (rounds_left*4 < my_bankroll/2):
        #     # always fold

        pass


    def update_bounty_prob(self, opp_hand, opp_bounty_hit):
        # Create a mapping from rank to index for easy access
        rank_to_index = {rank: idx for idx, rank in enumerate(ranks)}
        
        # Initialize a dictionary to count occurrences of each rank in opp_hand
        rank_counts = {rank: 0 for rank in ranks}
        for card in opp_hand:
            rank = ranks[card.rank]
            if rank in rank_counts:
                rank_counts[rank] += 1
        
        remaining_deck_size = 52 - len(opp_hand)
        unnormalized_probs = []
        
        for rank in ranks:
            idx = rank_to_index[rank]
            count_seen = rank_counts[rank]
            remaining_count = 4 - count_seen 
            
            if remaining_count < 0:
                remaining_count = 0
            
            if opp_bounty_hit:
                if count_seen > 0:
                    likelihood = 1.0
                else:
                    try:
                        combinations_total = math.comb(remaining_deck_size, 2)
                        combinations_without_rank = math.comb(remaining_deck_size - remaining_count, 2)
                        prob_no_rank = combinations_without_rank / combinations_total
                        likelihood = 1.0 - prob_no_rank
                    except ValueError:
                        likelihood = 0.0
            else:
                if count_seen > 0:
                    likelihood = 0.0
                else:
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
            return opp_bounty_prob.copy()
        
        # Normalize the probabilities so that they sum to 1
        updated_prob = np.array(unnormalized_probs, dtype=np.float64)
        updated_prob /= total_unnormalized

        # for rank, prob in zip(ranks, updated_prob):
        #     print(f"Rank {rank}: {prob:.2%}")
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
        
        # # update bankroll
        # global past_bankroll
        # past_bankroll += my_delta


    def generate_possible_hands(self, deck, my_hand):
        ls = []
        for card1 in deck:
            for card2 in deck:
                if not (card1 == card2 or card1 in my_hand or card2 in my_hand):
                    ls.append((card1, card2))
        return ls
    

    def weight_hands(self, hands, board_cards):
        weights = {}
        # keep track of rank and suit
        

        for hand in hands:
            # these are indices
            rank_count = [0] * 13
            suit_count = [0] * 4
            for cards in board_cards:
                rank_count[cards.rank] += 1
                suit_count[cards.suit] += 1
            rank1 = hand[0].rank
            rank2 = hand[1].rank
            suit1 = hand[0].suit
            suit2 = hand[1].suit
            rank_count[rank1] += 1
            rank_count[rank2] += 1
            suit_count[suit1] += 1
            suit_count[suit2] += 1

            weights[hand] = 1.0

            #pockets
            pocket = False
            if rank1 == rank2:
                weights[hand] = math.floor(1.4+rank1/12)
                pocket = True
            #quads
            if (rank_count[rank1] == 4 or rank_count[rank2] == 4):
                weights[hand] = 25.0
            trips = False
            tripIndex = -1
            #trips
            if (rank_count[rank1] == 3 or rank_count[rank2] == 3):
                weights[hand] = 4.5 if pocket else 4.0
                trips = True
                tripIndex = rank1 if rank_count[rank1] == 3 else rank2
            for card in board_cards:
                # full house
                if trips and (rank_count[card.rank] >= 2 and card.rank != tripIndex):
                    weights[hand] = 15.0
                    break
                #pair 
                if rank1 == card.rank:
                    weights[hand] *= math.floor(1.2+rank1/20)
                if rank2 == card.rank:
                    weights[hand] *= math.floor(1.2+rank1/20)

            # straight draw and flush draw
            flush = False
            for suit in range(4):
                if suit_count[suit] == 4:
                    weights[hand] *= 3.5 if len(board_cards) == 3 else 2.4
                elif suit_count[suit] == 5:
                    weights[hand] = 10.0
                    flush = True
            psums = []
            psums.append(0)
            for i in range(13):
                psums.append(psums[i] + min(1, rank_count[i]))
            for i in range(5,14):
                val = psums[i] - psums[i-5]
                if val == 4:
                    weights[hand] *= 1.6 if len(board_cards) == 3 else 1.3
                elif val == 5:
                    weights[hand] = 8.0 if not flush else 35.0
        return weights


    # FIGURE THESE WEIGHTS OUT, measure board wetness?
    def update_range_based_on_action(self, hands, weights, action):
        if action == 'raise':
            hands = [hand for hand in hands if weights[hand] >= 2.0]
        elif action == 'call':
            # Calling implies medium-strength hands
            hands = [hand for hand in hands if weights[hand] >= 1.5 ]

        # this shouldn't even occur because we don't care if they fold
        elif action == 'fold':
            hands = []  # No hands left
        return hands


    # def normalize_weights(self, weights):
    #     total_weight = sum(weights.values())
    #     for hand in weights:
    #         weights[hand] /= total_weight
    #     return weights


    def estimate_opponent_range(self, my_hand, board_cards, action):
        deck = val.Deck()
        hands = self.generate_possible_hands(deck, my_hand)
        weights = self.weight_hands(hands, board_cards)
        hands = self.update_range_based_on_action(hands, weights, action)
        weights = {hand: weights[hand] for hand in hands}  # Filter weights for remaining hands
        # normalized_weights = self.normalize_weights(weights)
        # return normalized_weights
        return weights


    def calculate_pot_odds(self, is_raise, my_contribution, opp_contribution, cost, my_bounty_hit, equity, opp_bounty_hit):
        
        # Compute expected_loss:
        # If opponent's bounty was hit and they win, their winnings are 1.5 * usual_winnings + 10
        # If bounty not hit and they win, their winnings are usual_winnings
        my_total = my_contribution+cost
        expected_loss = opp_bounty_hit * (1.5 * cost + 10) + (1.0 - opp_bounty_hit) * cost
        
        if not is_raise:
            if my_bounty_hit:
                ev = equity * (1.5 * opp_contribution + 10) - (1.0 - equity) * expected_loss
            else: 
                ev = equity * opp_contribution - (1.0 - equity) * expected_loss
        else: 
            if my_bounty_hit:
                ev = equity * (1.5 * my_total + 10) - (1.0 - equity) * expected_loss
            else: 
                ev = equity * my_total - (1.0 - equity) * expected_loss

        # Compute Pot Odds
        pot_odds = 1000  #check pot odds set to 1000 to always check if we should and are able to
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

        time = game_state.game_clock
        print(f"Action time: {time}")
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
        my_hand = [val.Card(s) for s in my_cards+board_cards]

        print(my_cards)
        print(board_cards)
        MY_CARDS = [val.Card(s) for s in my_cards]
        BOARD_CARDS = [val.Card(s) for s in board_cards]
        MY_HAND = MY_CARDS+BOARD_CARDS

        my_bounty_hit = False
        if my_bounty in my_hand:
            my_bounty_hit = True

        if opp_contribution > my_contribution: 
            action = 'raise'
        else:
            action = 'call'
        weights = self.estimate_opponent_range(MY_HAND, BOARD_CARDS, action)
        print(weights)

        equity, opp_bounty_hit = py_hand_vs_weighted_range_monte_carlo(MY_CARDS, weights, BOARD_CARDS, opp_bounty_prob, NUM_SIMULATIONS)
        print(f"Equity: {equity}")
        print(f"Opponent bounty hit: {opp_bounty_hit}")

        if CheckAction in legal_actions: 
            check_pot_odds = self.calculate_pot_odds(False, my_contribution, opp_contribution, continue_cost, my_bounty_hit, equity, opp_bounty_hit)
        if CallAction in legal_actions:
            call_pot_odds = self.calculate_pot_odds(False, my_contribution, opp_contribution, continue_cost, my_bounty_hit, equity, opp_bounty_hit)
            print(f"Call pot odds: {call_pot_odds:.3f}")
        if RaiseAction in legal_actions:
            all_in_pot_odds = self.calculate_pot_odds(True, my_contribution, opp_contribution, max_cost, my_bounty_hit, equity, opp_bounty_hit)
            print(f"All in pot odds: {all_in_pot_odds:.3f}")
        
        # !!!! CHANGE ALL NUMBERS TO FLOAT. Ex: 2 --> 2.0

        
        # ALL IN BOT
        if RaiseAction in legal_actions:
            if all_in_pot_odds >= 0:
                return RaiseAction(max_raise)
        if CallAction in legal_actions: 
            if call_pot_odds >= 0:
                return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        else: 
            return FoldAction()