'''
This file contains the base class that you should implement for your pokerbot.
'''

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7 as val
import itertools
from eval7 import py_hand_vs_range_monte_carlo

NUM_SIMULATIONS = 1_000_000

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
        pass

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
        my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty

    def generate_possible_hands(deck, n):
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n > len(deck.cards):
            raise ValueError(f"Cannot generate hands with {n} cards from a deck of {len(deck.cards)} cards.")

        # Use itertools.combinations to generate all possible n-card combinations
        # This returns tuples of eval7.Card objects
        ls = []
        for hand in itertools.combinations(deck.cards, n):
            temp = []
            for card in hand:
                temp.append(card)
                ls.append([temp])
        return ls


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

        deck = val.Deck()
        hand = my_cards+board_cards # idk if this is correct syntax
        bounty_hit = False
        if my_bounty in hand:
            bounty_hit = True
        eval = val.evaluate(hand)
        for card in my_cards:
            deck.cards.remove(card)

        # Define the opponent's range as all possible two-card combinations remaining in the deck
        opponent_range = Bot.generate_possible_hands(deck, 2)

        equity = py_hand_vs_range_monte_carlo(my_cards, opponent_range, board_cards, NUM_SIMULATIONS)
        # print(f"Your equity on the flop is approximately {equity:.2f}")

        board_total = my_pip+opp_pip
        if not bounty_hit: 
            pot_odds = (continue_cost)/(board_total+continue_cost)
            min_raise_pot_odds = (min_raise)/(board_total+min_raise)
            # add 3x raise, 6x raise, 10x raise etc. 
            all_in_pot_odds = (max_raise)/(board_total+max_raise)
        else: 
            pot_odds = (continue_cost)/(1.5*(board_total+continue_cost)+10)
            min_raise_pot_odds = (min_raise)/(1.5*(board_total+min_raise)+10)
            # add 3x raise, 6x raise, 10x raise etc. 
            all_in_pot_odds = (max_raise)/(1.5*(board_total+max_raise)+10)

        # somewhat gto bot

        # thresholds to include the fact that our opponent might have a good hand, variables subject to change
        all_in_threshold = 0.20
        min_raise_threshold = 0.10
        call_threshold = 0.03

        # raise equity
        if RaiseAction in legal_actions:
            if equity > all_in_pot_odds+all_in_threshold:
                return RaiseAction(max_raise)
            if equity > min_raise_pot_odds+min_raise_threshold:
                return RaiseAction(min_raise)
            
        # call equity
        if (equity > pot_odds+call_threshold and CallAction in legal_actions):
            return CallAction
        
        # check/fold
        if CheckAction in legal_actions: 
            return CheckAction
        else: 
            return FoldAction

        
        # #ALL IN BOT
        # if RaiseAction in legal_actions:
        #     return RaiseAction(max_raise)
        # if CallAction in legal_actions: 
        #     return CallAction
        # else:
        #     return CheckAction

        # if RaiseAction in legal_actions:
        #     if random.random() < 0.5:
        #         return RaiseAction(min_raise)
        # if CheckAction in legal_actions:  # check-call
        #     return CheckAction()
        # if random.random() < 0.25:
        #     return FoldAction()
        # return CallAction()
