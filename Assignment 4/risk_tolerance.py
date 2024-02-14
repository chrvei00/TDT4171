import numpy as np

def utility(x, R):
    return -(np.exp((-x)/R))

def a():
    print("\nPart A")
    R = 500
    guaranteed_return = 500
    risky_return = 5000
    chanche_risky = 0.6
    utility_guaranteed = utility(guaranteed_return, R)
    utility_risky = chanche_risky*utility(risky_return, R) + (1-chanche_risky)*utility(0, R)
    #Mary is rational and risk averse, so she will choose the highest expected utility
    # print(f"Utility of guaranteed return: {utility_guaranteed}")
    # print(f"Utility of risky return: {utility_risky}")
    if utility_guaranteed > utility_risky:
        print(f"Mary's choosen the guaranteed return of {guaranteed_return} with utility of {utility_guaranteed}\n")
        return guaranteed_return
    else:
        print(f"Mary's choosen the risky return of {risky_return} with utility of {utility_risky}\n")
        return risky_return

def b(initial_guess=100, epochs=1000, learning_rate=0.1):
    print("\nPart B")
    R = initial_guess
    guaranteed_return = 100
    risky_return = 500
    chanche_risky = 0.5
    for i in range(epochs):
        utility_guaranteed = utility(guaranteed_return, R)
        utility_risky = chanche_risky*utility(risky_return, R) + (1-chanche_risky)*utility(0, R)
        if utility_guaranteed > utility_risky:
            R += max(learning_rate, 0.1)
        else:
            R -= max(learning_rate, 0.1)

    print(f"R: {R}")
    print(f"Utility of guaranteed return: {utility_guaranteed}")
    print(f"Utility of risky return: {utility_risky}\n")

if __name__ == "__main__":
    a()
    b()
