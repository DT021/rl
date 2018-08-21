import numpy as np

def assess_card(card_string):
    if card_string == "A":
        return "A"
    if card_string == "J" or card_string == "Q" or card_string == "K":
        return 10
    else:
        return int(card_string)


def calculate_score(cards_list, return_minimum=False):
    soft = 0
    while "A" in cards_list:
        soft += 1
        cards_list.remove("A")

    min_score = np.sum([assess_card(card) for card in cards_list]) + soft

    if return_minimum:
        return min_score

    if soft == 1:
        possible_scores = [min_score, min_score + 10]
    if soft >= 2:
        possible_scores = [min_score, min_score + 10]
    else:
        possible_scores = [min_score]
    final_score = min_score
    for score in possible_scores:
        if score > final_score and score <= 21:
            final_score = score
    return final_score