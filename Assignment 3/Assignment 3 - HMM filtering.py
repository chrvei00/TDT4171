import numpy as np

def run_HMM_filter_forward(initial_state, transition_matrix, emission_matrix, observations):
    forward = np.zeros((len(initial_state), len(observations)))
    forward[:, 0] = initial_state * emission_matrix[:, observations[0]]
    for i in range(1, len(observations)):
        forward[:, i] = np.dot(forward[:, i-1], transition_matrix) * emission_matrix[:, observations[i]]
    prob = forward[:, -1] / np.sum(forward[:, -1])
    return prob


def main():
    initial_state = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    observations = np.array([0, 0])
    prob = run_HMM_filter_forward(initial_state, transition_matrix, emission_matrix, observations)
    print(f"\nInitial state: \n{initial_state}")
    print(f"\nTransition matrix: \n{transition_matrix}")
    print(f"\nEmission matrix: \n{emission_matrix}")
    print(f"\nObservations: {observations}")
    print(f"\nP(rain|observations) = {prob[0]:.4f}")
    print(f"P(sun|observations) = {prob[1]:.4f}\n")

if __name__ == "__main__":
    main()