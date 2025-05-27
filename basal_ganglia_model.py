import numpy as np

class BasalGangliaModel:
    def __init__(self, dopamine=0.5, n_actions=2):
        """
        Simplified basal ganglia model based on the provided repository.
        
        Parameters:
        - dopamine: dopamine level (0-1)
        - n_actions: number of possible actions
        """
        self.dopamine = np.clip(dopamine, 0, 1)
        self.n_actions = n_actions
        
        # Initialize weights for the different pathways
        # These values are simplified approximations
        self.direct_weights = np.array([0.7, 0.7])  # Facilitates action selection
        self.indirect_weights = np.array([0.5, 0.5])  # Suppresses competing actions
        self.hyperdirect_weights = np.array([0.3, 0.3])  # Rapid action suppression
        
        # Dopamine modulation factors
        self.da_mod_direct = 1.0 + 0.5 * self.dopamine  # DA enhances direct pathway
        self.da_mod_indirect = 1.0 - 0.5 * self.dopamine  # DA suppresses indirect pathway
        
    def process(self, q_values):
        """
        Process Q-values through basal ganglia pathways to produce action probabilities.
        
        Parameters:
        - q_values: array of Q-values for each action
        
        Returns:
        - action_probs: probabilities for each action
        """
        q_values = np.array(q_values)
        
        # Direct pathway (Go): promotes selected action
        direct = q_values * self.direct_weights[:len(q_values)] * self.da_mod_direct
        
        # Indirect pathway (NoGo): suppresses non-selected actions
        indirect = (1 - q_values) * self.indirect_weights[:len(q_values)] * self.da_mod_indirect
        
        # Hyperdirect pathway: global suppression
        hyperdirect = np.ones_like(q_values) * self.hyperdirect_weights[:len(q_values)]
        
        # Combine pathways
        net = direct - indirect - hyperdirect
        
        # Softmax to get probabilities
        exp_net = np.exp(net - np.max(net))  # For numerical stability
        action_probs = exp_net / np.sum(exp_net)
        
        return action_probs 

