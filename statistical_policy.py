from blackjack_policy import BlackjackPlayer
from blackjack_utils import calculate_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class StatisticalPlayer(BlackjackPlayer):
    def create_model(self):
        model = RandomForestRegressor(n_estimators=100)
        self.model = model

    def experience_memories(self, num_recent=None, batch_size=16, batch_count=16):
        # TODO: convert memories into a training set
        xs = []
        ys = []

        if num_recent:
            if len(self.memory) > num_recent:
                mems = self.memory[-num_recent:]
            else:
                mems = self.memory
        else:
            mems = self.memory

        mems_batch = np.random.choice(mems, batch_size, replace=False)
        for memory in mems_batch:
            x, y = self.convert_memory(memory)
            xs.append(x)
            ys.append(y)

        x = np.array(xs).reshape([len(xs), self.input_size])
        y = np.array(ys).reshape([len(ys), 1])

        x = self.scaler.transform(x)

        self.model.fit(X=x, y=y.ravel())

