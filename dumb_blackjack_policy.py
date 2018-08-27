from blackjack_policy import BlackjackPlayer
from blackjack_utils import calculate_score

class DumbPlayer(BlackjackPlayer):
    def run_policy(self, my_hand, seen_cards, dealer_hand, return_decision=True):
        return 0

    def experience_memories(self, num_recent=None, batch_size=16, batch_count=16):
        pass


class NovicePlayer(BlackjackPlayer):
    def run_policy(self, my_hand, seen_cards, dealer_hand, return_decision=True):
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

    def experience_memories(self, num_recent=None, batch_size=16, batch_count=16):
        pass