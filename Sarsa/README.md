# Sarsa Implementation

## GOAL: Implement Gradient-Descenet Sarsa with Linear Function Approximation

### Notes from paper:

Sarsa is a widely used temporal difference method that stochastically approximates the optimal value function based on the state transition and the reward samplles received from online interaction with thee environment.

The Original Sarsa Algorithm works as follows:
* At time step t, the agent is in state st.
* It then chooses an action at from a greedy policy based on its current estimate of the optimal value function Qt.
* Consequently, the agent receives a reward rt and the evironment transitions to state st+1. 
* The agent then chooses action at+1 based on the greedy policy.
* The following temporal difference update is used to update the estimate of the action-value function based on the given (st, at, rt, st+1, at+1) sample: Qt+1(st, at) = Qt(st, at) + αδ where δ = rt + γQt(st+1, at+1) − Qt(st, at) and α is the learning rate.


Sarsa(λ) extends the original Sarsa algorithm by introducing the idea of eligibility traces. Eligibility traces allow us to update the action-value function not only for the latest  (st, at) state-action pair, but also for all the recently visited state-actions. To do this, an additional eligibility value is stored for every state-action pair, denoted by e(s, a). At each step, the eligibility traces of all the state-action pairs are reduced by a factor of γλ. The eligibility trace of the latest state action n e(st, at) is then set to one

After each (st, at, rt, st+1, at+1) sample, δ is calculated the same as in equation 2.2. However, instead of only updating Q(st, at), all state-action pairs are updated based on their eligibility:

Qt+1(s, a) = Qt(s, a) + αδet(s, a) for all s,a

When doing linear function approximation, instead of keeping an eligibility value for each state, the eligibility trace vector et keeps an eligibility trace for each item in the feature vector. On each time step, after reducing the eligibility vector by a factor of γλ, the items corresponding to one-indices in the current feature vector are set to one. Additionally, we may wish to clear all the traces corresponding to all actions other than the current action at in state st

## RAM Implementation
Goal: Use RAM implementation to help with the screenshot pixel implementation

Note: Need python3 and a couple of dependencies but I didn't make a dependency file :(
* *To Run:* ```$ python sarsa0.py <Number of Games>```
    * Number Of Games: The number of games that you want the Agent to run through

* *Result:* At the end of the program a graph of the results will print



