def my_function(name, age, current_year):
    years_to = 100 - age
    ans=  current_year + years_to
    return ans


def main():  # Wrapper function
    name = input ("What is your name: ")
    age = int(input ("Enter the age in figures: "))
    current_year = int(input("Enter the current year in figures: "))
    years_to = 100 - age
    ans = current_year + years_to

    print(my_function(name, age, current_year))
    print("Hi " + str(name) +"! You will turn 100 in the year " + str(ans) )

if __name__ == "__main__":
    main()
