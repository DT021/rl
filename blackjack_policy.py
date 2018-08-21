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
        model.add(Dense(10, input_shape=(17, ), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(sgd(lr=.2), "mse")
        self.model = model

    def run_policy(self, my_hand, seen_cards, dealer_hand):
        # transform inputs
        if calculate_score(my_hand) < 16:
            final_output = 1
        else:
            final_output = 0

        # run model
        self.model.predict(x=np.ones(17).reshape(1, -1))

        # transform outputs
        output = final_output

        return output

