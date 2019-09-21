from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for j in range(L):
            if j == 0:
                for i in range(S):
                    alpha[i][j] = self.pi[i] * self.B[i][self.obs_dict[Osequence[j]]]
            else:
                for i in range(S):
                    for m in range(S):
                        alpha[i][j] += alpha[m][j-1] * self.A[m][i]
                    alpha[i][j] *= self.B[i][self.obs_dict[Osequence[j]]]
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        for j in range(L-1,-1,-1):
            if j == L-1:
                for i in range(S):
                    beta[i][j] = 1
            else:
                for i in range(S):
                    for m in range(S):
                        beta[i][j] += self.A[i][m] * self.B[m][self.obs_dict[Osequence[j+1]]] * beta[m][j+1]
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        for i in range(alpha.shape[0]):
            prob += alpha[i][alpha.shape[1]-1]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sprob = self.sequence_prob(Osequence)
        prob = alpha * beta / sprob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sprob = self.sequence_prob(Osequence)
        for i in range(S):
            for j in range(S):
                for t in range(L-1):
                    prob[i][j][t] = alpha[i][t] * self.A[i][j] * self.B[j][self.obs_dict[Osequence[t+1]]] * beta[j][t+1]
                    # prob[i][j][t] = self.A[i][j] * alpha[i][t] * self.B[j][self.obs_dict[Osequence[t+1]]] * beta[j][self.obs_dict[Osequence[t+1]]]
        prob /= sprob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)

        # new observation
        obs_list = list(self.obs_dict.keys())
        index = len(obs_list)
        for obs in Osequence:
            if obs not in obs_list:
                self.obs_dict[obs] = index
                self.B = np.column_stack((self.B,np.zeros(S)))
                for j in range(S):
                    self.B[j][index] = 1e-6
                index += 1

        sigma = np.zeros([S, L])
        for j in range(L):
            if j == 0:
                for i in range(S):
                    sigma[i][j] = self.pi[i] * self.B[i][self.obs_dict[Osequence[j]]]
            else:
                for i in range(S):
                    max = 0
                    for m in range(S):
                        if max < self.A[m][i] * sigma[m][j-1]:
                            max = self.A[m][i] * sigma[m][j-1]
                    sigma[i][j] = max * self.B[i][self.obs_dict[Osequence[j]]]

        for j in range(L-1,0,-1):
            value = np.argmax(sigma, axis=0)
            index = value[j]
            for i in range(S):
                sigma[i][j-1] *= self.A[i][index]

        value = np.argmax(sigma, axis=0)

        keys = self.state_dict.keys()
        for j in range(L):
            for key in keys:
                if self.state_dict[key] == value[j]:
                    path.append(key)
        ###################################################
        return path


    def modify(self,Osequence):
        S = len(self.pi)

        # new observation
        obs_list = list(self.obs_dict.keys())
        index = len(obs_list)
        for obs in Osequence:
            if obs not in obs_list:
                self.obs_dict[obs] = index
                self.B = np.column_stack((self.B,np.zeros(S)))
                for j in range(S):
                    self.B[j][index] = 1e-6
                index += 1