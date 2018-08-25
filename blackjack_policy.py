from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from blackjack_utils import assess_card, calculate_score
import numpy as np

class BlackjackPlayer():
    def __init__(self):
        self.model = None
        self.memory = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(18, ), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.compile(sgd(lr=.2), "mse")
        self.model = model

    def transform_inputs(self, my_hand, seen_cards, dealer_hand):
        my_score = [calculate_score(my_hand, return_minimum=True)]
        my_aces = [my_hand.count("A")]
        dealer_score = [calculate_score(dealer_hand, return_minimum=True)]
        dealer_aces = [dealer_hand.count("A")]

        cards_seen = []
        for card in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
            cards_seen.append(seen_cards.count(card))

        return cards_seen + my_score + my_aces + dealer_score + dealer_aces

    def run_policy(self, my_hand, seen_cards, dealer_hand, exploration=0.00, deterministic=True):
        # transform inputs
        inputs = self.transform_inputs(my_hand, seen_cards, dealer_hand)

        input_hit = inputs + [1]
        input_stay = inputs + [0]

        # run model
        final_output_hit = self.model.predict(x=np.array(input_hit).reshape(1, -1))[0]
        final_output_stay = self.model.predict(x=np.array(input_hit).reshape(1, -1))[0]

        # TODO: determine probability of the outputs for hit and stay
        final_output = self.move_scaler([final_output_stay, final_output_hit], min=-1, max=1)

        if deterministic:
            selection = np.argmax(final_output)
            return selection
        print("final_output: ", final_output)
        if np.random.uniform(0, 1) < exploration:
            return np.random.choice([i for i in range(len(final_output))])

        # transform outputs
        rand = np.random.uniform(0, 1)
        total = 0
        for selection in range(len(final_output)):
            total += final_output[selection]
            if rand < total:
                print("selection: ", selection)
                return selection

    def create_state(self, my_hand, seen_cards, dealer_hand):
        return {"my_hand": my_hand, "seen_cards": seen_cards, "dealer_hand": dealer_hand}

    def memorize(self, events):
        for event in events:
            self.memory.append({"state": self.create_state(event["my_hand"], event["seen_cards"], event["dealer_hand"]),
                                "action": event["action"], "reward": event["reward"]})

    def experience_memories(self, num_recent=None):
        # TODO: convert memories into a training set
        xs = []
        ys = []

        if num_recent:
            mems = self.memory[-num_recent:]
        else:
            mems = self.memory

        for memory in mems:
            x, y = self.convert_memory(memory)
            xs.append(x)
            ys.append(y)

        x = np.array(xs).reshape([len(xs), ])
        y = np.array(ys).reshape([len(ys), 1])
        # TODO: batch train with those memories


    def convert_memory(self, memory):
        # TODO: take one memory, convert it into x and y
        state = memory["state"]
        action = memory["action"]
        reward = memory["reward"]

        x = self.transform_inputs(state["my_hand"], seen_cards=state["seen_cards"], dealer_hand=state["dealer_hand"])
        x = x + [action]
        y = reward
        return x, y

    @staticmethod
    def move_scaler(x, min, max):
        scale = max - min
        scaled_x = [(v - min) / scale for v in x]

        total = np.sum(scaled_x)
        result = [v / total for v in scaled_x]
        return result