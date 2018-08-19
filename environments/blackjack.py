from environments.base import Environment
from random import shuffle
import numpy as np

class CardDeck:
    def __init__(self):
        self.available_cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4
        shuffle(self.available_cards)
    def draw_card(self):
        return self.available_cards.pop()

class Blackjack(Environment):
    def __init__(self, num_other_players=5):
        super(Environment, self).__init__()
        self.seen_cards = []
        self.my_hand = []
        self.dealer_hand = []
        self.deck = CardDeck()
        self.num_other_players = 5

    def assess_card(self, card_string):
        if card_string == "A":
            return "A"
        if card_string == "J" or card_string == "Q" or card_string == "K":
            return 10
        else:
            return int(card_string)

    def calculate_score(self, cards_list, return_minimum=False):
        soft = 0
        while "A" in cards_list:
            soft += 1
            cards_list.remove("A")

        min_score = np.sum([self.assess_card(card) for card in cards_list]) + soft
        
        if return_minimum:
            return min_score

        if soft == 1:
            possible_scores = [min_score, min_score+10]
        if soft >= 2:
            possible_scores = [min_score, min_score+10]
        else:
            possible_scores = [min_score]
        final_score = min_score
        for score in possible_scores:
            if score > final_score and score <= 21:
                final_score = score
        return final_score

    def default_policy(self, cur_score):
        if cur_score < 16:
            return 1
        else:
            return 0

    def dealer_draw(self):
        card = self.deck.draw_card()
        self.seen_cards.append(card)
        self.dealer_hand.append(card)

    def other_player_draw(self):
        card = self.deck.draw_card()
        self.seen_cards.append(card)
        return card

    def my_draw(self):
        card = self.deck.draw_card()
        self.seen_cards.append(card)
        self.my_hand.append(card)

    def run_policy(self, policy):
        pass
        # initialize hand


        # inputs are number of each card seen, my minimum hand score, how many aces I have
        # run policy repeatedly until policy returns either 0 (stay) or player score exceeds 21

    def initialize_hand(self):
        # distribute cards to other players
        for other_player in range(self.num_other_players):
            other_player_hand = []
            other_player_score = 0
            while other_player_score < 21:
                if self.default_policy(other_player_score):
                    other_player_hand.append(self.other_player_draw())
                    other_player_score = self.calculate_score(other_player_hand)
                else:
                    break

