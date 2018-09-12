from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt

# Function to generate one (1) sample random walk sequence
def generate_random_walk():
    sequence = [3]
    states = np.array(['A', 'B', 'C','D', 'E', 'F', 'G'])
    states_x = {'A': 0,
                'B': np.array([1, 0, 0, 0, 0]),
                'C': np.array([0, 1, 0, 0, 0]),
                'D': np.array([0, 0, 1, 0, 0]),
                'E': np.array([0, 0, 0, 1, 0]),
                'F': np.array([0, 0, 0, 0, 1]),
                'G': 1
                }
    walk = 1
    while walk:
        if np.random.random() <= 0.5:
            if sequence[-1] - 1 == 0:
                sequence.append(sequence[-1] - 1)
                walk = 0
            else:
                sequence.append(sequence[-1] - 1)
        else:
            if sequence[-1] + 1 == 6:
                sequence.append(sequence[-1] + 1)
                walk = 0
            else:
                sequence.append(sequence[-1] + 1)
    return (states[sequence], np.array([states_x[x] for x in states[sequence]]))

# Generate 100*10 sequences
np.random.seed(123)
training_sets = []
for i in range(100):
    training_set = []
    for j in range(10):
        training_set.append(generate_random_walk())
    training_sets.append(training_set)
training_sets = np.array(training_sets)



############ Figure 3 Implementation ############
## TD(lambda)
lambdas = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
alpha = 0.01
true_value = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
rms_all = []
for l in lambdas:  # Loop all lambdas
    rms_sets = []
    for i in range(100):    # Loop 100 training sets
        # Iterate to update w until converge
        w = np.zeros(5)
        while True:
            delta_w = np.zeros(5)
            for j in range(10):     # Loop 10 sequences in each training set
                num_states = training_sets[i][j][0].size - 1
                P = [training_sets[i][j][1][0].dot(w)]
                e = np.zeros(5)
                for k in range(num_states):     # Loop all states
                    # Calculate P(t+1)
                    if k != num_states - 1:
                        P.append(training_sets[i][j][1][k+1].dot(w))
                    else:
                        P.append(training_sets[i][j][1][k+1])
                    # Update e
                    e = l * e + training_sets[i][j][1][k]
                    # Add up delta(W)
                    delta_w += alpha * (P[k+1] - P[k]) * e
            # Check if converge
            if sum(delta_w) > 0.001:
                w += delta_w
            else:
                break
        # Calculate RMS for each sequence
        rms_set = []
        for j in range(10):
            rms_set.append(np.sqrt(sum((w - true_value)**2)/5))
        rms_set = sum(rms_set)/len(rms_set)
        # Add RMS of the training set to the training sets list
        rms_sets.append(rms_set)
    # Average RMS for the 100 training sets
    rms_sets = sum(rms_sets)/len(rms_sets)
    rms_all.append(rms_sets)

## Plot RMS
plt.plot(lambdas, rms_all, 'o-')
plt.xlabel('Lambda')
plt.ylabel('RMS Error')
plt.title('Figure 3')
plt.xticks(lambdas)
plt.text(0.8, rms_all[-1], 'Widrow-Hoff')
plt.show()


############ Figure 4 Implementation ############
lambdas = np.array([0.0, 0.3, 0.8, 1.0])
alphas = np.arange(0, 0.55, 0.05)
true_value = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
rms_all = []
for l in lambdas:  # Loop all lambdas
    rms_alpha = []
    for a in alphas:
        rms_sets = []
        for i in range(100):    # Loop 100 training sets
            w = np.ones(5) * 0.5
            for j in range(10):     # Loop 10 sequences in each training set
                delta_w = np.zeros(5)
                num_states = training_sets[i][j][0].size - 1
                P = [training_sets[i][j][1][0].dot(w)]
                e = np.zeros(5)
                for k in range(num_states):     # Loop all states
                    # Calculate P(t+1)
                    if k != num_states - 1:
                        P.append(training_sets[i][j][1][k+1].dot(w))
                    else:
                        P.append(training_sets[i][j][1][k+1])
                    # Update e
                    e = l * e + training_sets[i][j][1][k]
                    # Add up delta(W)
                    delta_w += a * (P[k+1] - P[k]) * e
                # Update w
                w += delta_w
                # Calculate RMS for each sequence
                rms_sets.append(np.sqrt(sum((w - true_value) ** 2) / 5))
        # All average RMS of all 100*10 sequences
        rms_sets = sum(rms_sets)/len(rms_sets)
        rms_alpha.append(rms_sets)
    rms_all.append(rms_alpha)

## Plot RMS
for i in range(4):
    plt.plot(alphas, rms_all[i], 'o-')
plt.xlabel('Alpha')
plt.ylabel('RMS Error')
plt.title('Figure 4')
plt.xticks(alphas)
plt.legend(lambdas)
plt.text(0.47, rms_all[0][-1], '0.0')
plt.text(0.47, rms_all[1][-1], '0.3')
plt.text(0.47, rms_all[2][-1], '0.8')
plt.text(0.3, rms_all[3][-1], 'lambda = 1.0 (Widrow-Hoff)')
plt.show()



############ Figure 5 Implementation ############
# Algorithm similar to figure 4, but with more lambda
lambdas = np.arange(0, 1.1, 0.1)
alphas = np.arange(0, 0.55, 0.05)
true_value = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
rms_all = []
for l in lambdas:  # Loop all lambdas
    rms_alpha = []
    for a in alphas:
        rms_sets = []
        for i in range(100):    # Loop 100 training sets
            w = np.ones(5) * 0.5
            for j in range(10):     # Loop 10 sequences in each training set
                delta_w = np.zeros(5)
                num_states = training_sets[i][j][0].size - 1
                P = [training_sets[i][j][1][0].dot(w)]
                e = np.zeros(5)
                for k in range(num_states):     # Loop all states
                    # Calculate P(t+1)
                    if k != num_states - 1:
                        P.append(training_sets[i][j][1][k+1].dot(w))
                    else:
                        P.append(training_sets[i][j][1][k+1])
                    # Update e
                    e = l * e + training_sets[i][j][1][k]
                    # Add up delta(W)
                    delta_w += a * (P[k+1] - P[k]) * e
                # Update w
                w += delta_w
                # Calculate RMS for each sequence
                rms_sets.append(np.sqrt(sum((w - true_value) ** 2) / 5))
        # All average RMS of all 100*10 sequences
        rms_sets = sum(rms_sets)/len(rms_sets)
        rms_alpha.append(rms_sets)
    rms_all.append(rms_alpha)

rms_all = np.array(rms_all)
rms_min = np.min(rms_all, 1)

## Plot RMS
plt.plot(lambdas, rms_min, 'o-')
plt.xlabel('Lambda')
plt.ylabel('RMS Error')
plt.title('Figure 5')
plt.xticks(lambdas)
plt.text(0.8, rms_min[-1], 'Widrow-Hoff')
plt.show()