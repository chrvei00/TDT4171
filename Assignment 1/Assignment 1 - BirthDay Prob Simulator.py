import random

def birthdaySimulator(numOfPeople):
    birthdays = []
    for i in range(numOfPeople):
        birthdays.append(random.randint(1, 365))
    
    # Check if there are any duplicate birthdays
    if len(birthdays) != len(set(birthdays)):
        return True
    else:
        return False
    

def part1():
    #Variables
    N = 23
    simulations = 100000

    #Simulate
    duplicates = 0
    for i in range(simulations):
        if birthdaySimulator(N):
            duplicates += 1
    print("\nThe probability in a group of " + str(N) + " is " + str(duplicates / simulations) + ".\n")

def fillAllBirthdays():
    counter = 0
    birthdays = []
    while len(birthdays) != 365:
        counter += 1
        tmp_birthday = random.randint(1, 365)
        if tmp_birthday not in birthdays:
            birthdays.append(random.randint(1, 365))
    return counter

def part2():
    #Variables
    simulations = 10000

    #Simulate
    N = 0
    for i in range (simulations):
        N += fillAllBirthdays()

    print("\nThe expected N is " + str(N / simulations) + ".\n")


def main():
    part1()
    #part2()

if __name__ == "__main__":
    main()