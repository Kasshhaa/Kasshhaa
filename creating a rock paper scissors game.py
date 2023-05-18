def game (player1,player2):
    if player1 not in "rock, paper, scissors":
        return "Invalid! Please type one of these; rock, paper or scissors"
    if player2 not in "rock, paper, scissors":
        return "Please type one of these rock, paper or scissors"

    if player1==player2:
        return "A tie, try again!"

    elif player1 == "rock" :
        if player2 =="scissors":
            return ("Congratulations! Player 1 wins")
        else:
            return ("Congratulations! Player 2 wins")

    elif player1 == "scissors":
        if player2 == "paper":
            return ("Congratulations! Player 1 wins")
        else :
            return("Congratulations! Player 2 wins")

    elif player1 == "paper":
        if player2 == "rock":
            return ("Congratulations! Player 1 wins")
        else :
            return ("Congratulations! Player 2 wins")



def main():  # Wrapper function

    player1 = input("Player 1, choose between rock, paper and scissors: ")
    player2 = input("Player 2, choose between rock, paper and scissors: ")


    print(game(player1, player2))

if __name__ == '__main__':
    main()
