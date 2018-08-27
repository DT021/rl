from environments.base import Environment
from random import shuffle
from blackjack_utils import assess_card, calculate_score
import numpy as np
from blackjack_policy import BlackjackPlayer
from dumb_blackjack_policy import DumbPlayer, NovicePlayer
from statistical_policy import StatisticalPlayer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


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
        self.num_other_players = 10


    def default_policy(self, cur_score):
        if cur_score < 16:
            return 1
        else:
            return 0

    def naiive_policy(self, my_hand, seen_cards, dealer_hand):
        cur_score = calculate_score(my_hand)
        if cur_score < 16:
            return 1
        elif cur_score > 16:
            return 0
        else:
            if "A" in my_hand:
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

    def dealer_play(self):
        dealer_score = calculate_score(self.dealer_hand, return_minimum=True)
        decision = 1
        while dealer_score < 17 and decision:
            self.dealer_draw()
            decision = self.naiive_policy(self.dealer_hand, self.seen_cards, self.dealer_hand)
        return calculate_score(self.dealer_hand)

    def play_hand(self, player):
        # deal cards and let all other players play
        self.initialize_hand()

        events = []
        # inputs are number of each card seen, my minimum hand score, how many aces I have
        # run policy repeatedly until policy returns either 0 (stay) or player score exceeds 21
        policy_result = 1
        score = 0
        while policy_result and calculate_score(self.my_hand) < 21:
            self.my_draw()
            policy_result = player.run_policy(self.my_hand, self.seen_cards, self.dealer_hand)
            events.append({"my_hand": list(self.my_hand),
                           "seen_cards": list(self.seen_cards),
                           "dealer_hand": list(self.dealer_hand),
                           "action": int(policy_result)})
            score = calculate_score(self.my_hand)

        dealer_score = self.dealer_play()

        if score == 21:
            reward = 1
        elif score > 21:
            reward = -1
        elif dealer_score > 21:
            reward = 1
        elif score > dealer_score:
            reward = 1
        elif score == dealer_score:
            reward = 0
        else:
            reward = -1

        for event in events:
            event["reward"] = reward

        player.memorize(events)

        return reward


    def run_policy(self, policy, **kwargs):
        return policy(kwargs)


    def initialize_hand(self):
        self.deck = CardDeck()
        self.seen_cards = []
        self.my_hand = []
        self.dealer_hand = []
        # distribute cards to other players
        for other_player in range(self.num_other_players):
            other_player_hand = []
            other_player_score = 0
            while other_player_score < 21:
                if self.default_policy(other_player_score):
                    other_player_hand.append(self.other_player_draw())
                    other_player_score = calculate_score(other_player_hand)
                else:
                    break
        self.dealer_draw()
        self.my_draw()

if __name__ == "__main__":
    bj_env = Blackjack()
    bj_player = BlackjackPlayer()
    bj_player.create_model()
    bj_player.deterministic = True
    bj_player.exploration = 1.0

    batch_size = 1024
    batch_count = 100
    hands_per_generation = 20000  # 10 thousand
    num_memories = 10000  # 100 thousand
    generations = 100


    all_results = []
    moving_results = []
    for generation in range(generations):  # for each generation
        gen_results = []
        for i in range(hands_per_generation):    # play new hands and gather experience
            result = bj_env.play_hand(bj_player)
            gen_results.append(result)
            all_results.append(result)

        # learn based on recent experiences
        bj_player.experience_memories(num_recent=num_memories,  # don't pay attention to old irrelevant memories
                                      batch_size=batch_size,    # how big each batch is
                                      batch_count=batch_count)   # how many batches to do

        generation_mean = np.mean(gen_results)
        print(f"generation {generation+1} score: {generation_mean}")

        bj_player.exploration = bj_player.exploration*.95  # on the next generation be less adventurous

    # test results
    test_results = []
    test_size = 10000 # 10 thousand
    bj_player.exploration = 0.0
    for i in range(test_size):
        result = bj_env.play_hand(bj_player)
        test_results.append(result)

    print(f"test_results: {np.mean(test_results)}")


