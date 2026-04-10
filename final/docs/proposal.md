# DS4400 Final Project Proposal

## Modeling and Predicting Player Actions in Low-Stakes Poker

## Team Members
- Luca Miniati
- Chigozirim Ike

## Problem Description
- How do people play the game of No-Limit Texas Holdem (NLH)? What motivates
their choices of actions?
- Given information about the current game state (e.g., hand strength,
position, pot size, stack depth), the goal is to predict the player’s next
action (fold, call, or raise).
- From a machine learning perspective, poker provides a rich, noisy, and
strategic environment with imperfect information, making it a strong testbed
for classification models.
- Accurate models of low-stakes player behavior can be used to study common
strategic mistakes, build exploitative agents, and better understand human
decision patterns under uncertainty. Low-stakes games are especially
interesting because players often follow simple or inconsistent heuristics,
which allows comparison between linear and highly flexible models.

## Dataset
- Source: https://poker.cs.ualberta.ca/IRC/IRCdata.tgz (IRC Poker Database)
- The dataset contains NLH hand histories with features describing hole cards,
community cards, and action history.
- The pipeline produces ~444,304 decision points (fold/call/raise moments) from the NLH hand histories.
- We identified 29 relevant features across five groups: draw, board texture, pot/betting, action, and game state.

## Approach and methodology
- Feature selection
    - Draw Features
        - Flush draw indicator
        - Straight draw indicator
    - Board Texture Features
        - Number of suited cards on board
        - Highest run of connected cards
        - Board pair indicator
        - Board trips indicator
        - Monotone board flag
        - Two-tone board flag
        - Rainbow board flag
        - Highest board rank
    - Pot/Betting Features
        - Current pot size (BB)
        - Effective stack (BB)
        - Stack-to-pot ratio (SPR)
        - Pot odds
        - Facing bet size (BB)
        - Facing bet as % of pot
        - All-in indicator
        - Commitment level (% of stack committed)
        - Minimum Defense Frequency (MDF)
    - Action Features
        - Count of aggressive actions made by opponent
        - Count of passive actions made by opponent
        - Last action (index 1)
        - Last action (index 2)
        - Villain check-raise indicator
        - Villain donk bet indicator
    - Game State Features
        - Hero Position index
        - Villain Position index
        - Preflop aggressor indicator
        - Street index
- The main task will be to predict call/fold/raise (3-class multinomial classification for all models)
- Machine learning models
    - Logistic Regression
    - Multinomial Logistic Regression
    - Random Forest Classifier
    - Gradient Boosted Trees
    - Recurrent Neural Network
- Python Packages
    - pokerkit (package)
    - sklearn
    - torch
- Evaluation
    - F1 score
    - Confusion Matrix

## Outcome
- We will try to find accurate ways to model opponent behavior.
- When applicable, we hope to perform inference, providing insights on the
factors that motivate different actions in poker.

## Plan
- Data/Preprocessing
    - Luca
- Training of models
    - Logistic Regression
        - Luca
    - Multinomial Logistic Regression
        - Chigo
    - Random Forest Classifier
        - Chigo
    - Gradient Boosted Trees
        - Luca
    - Recurrent Neural Network
        - Chigo
- Evaluation/Inference
    - Luca/Chigo
