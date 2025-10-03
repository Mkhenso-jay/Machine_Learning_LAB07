# run_all.py

##% Import Libraries
import os
from part1_iris import main as part1_main
from part2_bagging import main as part2_main
from part3_adaboost import main as part3_main
from part4_comparison import main as part4_main

##% Main Function
def main():
    ##% Create Outputs Directory
    os.makedirs("outputs", exist_ok=True)

    ##% Run Part 1: Majority Voting
    print("Running Part 1: Majority Voting on Iris Dataset")
    part1_main()

    ##% Run Part 2: Bagging
    print("\nRunning Part 2: Bagging on Wine Dataset")
    part2_main()

    ##% Run Part 3: AdaBoost
    print("\nRunning Part 3: AdaBoost on Wine Dataset")
    part3_main()

    ##% Run Part 4: Comprehensive Comparison
    print("\nRunning Part 4: Comprehensive Comparison")
    part4_main()

if __name__ == "__main__":
    main()