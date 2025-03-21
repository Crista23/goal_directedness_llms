These scripts can be run after experiment results have been stored in a 'results' folder in the main directory.

First run expected_return.py, to run Monte Carlo simulations for the return we would expect from an agent that fully used its capabilities, as well as for baseline performance.

This creates a file expected_return_X.csv, where X represents the number of iterations. By default it's set to 10, but for actual experiments we recommend setting num_iterations to at least 1000.

Once expected_return_10.csv exists, you can can run 

python3 goal_directedness.py 

to compute the goal-directedness of each model. The full data is stored in goal_directedness.csv, while the main takeaways are printed.
