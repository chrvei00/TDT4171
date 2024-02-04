import random

def slot(bet):
    row1 = random.randint(1, 4)
    row2 = random.randint(1, 4)
    row3 = random.randint(1, 4)
    # print("\nYou rolled: ")
    # print(row1, row2, row3)
    if row1 == row2 == row3:
        if row1 == 1:
            return bet * 20
        elif row1 == 2:
            return bet * 15
        elif row1 == 3:
            return bet * 5
        elif row1 == 4:
            return bet * 3
    elif row1 == 4 and row2 == 4:
        return bet * 2
    elif row1 == 4:
        return bet
    else:
        return 0

def playSlots(money, bet):
    counter = 0
    peak = money
    while(money >= bet):
        counter += 1
        # print("\nRound " + str(counter) + ":")
        # print("You have $" + str(money) + ".")
        money -= bet
        money += slot(bet)
        if money > peak:
            peak = money
    # print("\nYou have no more money to play with. \nYou made it " + str(counter) + " rounds. \nYour peak was $" + str(peak) + "")
    return (counter, peak)





def main():
    #Variables
    bet = 1
    money = 10
    numOfRounds = 100000
    
    #Play the game
    maxPeak = 0
    meanRounds = 0
    meanPeak = 0
    for i in range(1, numOfRounds + 1):
        res = playSlots(money, bet)
        meanRounds += res[0]
        meanPeak += res[1]
        maxPeak = max(maxPeak, res[1])
    # print("\n")
    meanRounds = meanRounds / numOfRounds
    meanPeak = meanPeak / numOfRounds
    print("The mean number of rounds was " + str(meanRounds) + ".")
    print("The mean peak was $" + str(meanPeak) + ".")
    print("The max peak was $" + str(maxPeak) + ".")

if __name__ == "__main__":
    main()