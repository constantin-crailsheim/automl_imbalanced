data_ids = (976, 980, 1002, 1018, 1019, 1021, 1040, 1053, 1461, 41160)
scoring = "balanced_accuracy"

# Set output path
output_path = "results/run_2"

# Set total maximum cost of optimisation in seconds.
total_cost = 30

# Set number of CV splits.
outer_cv_folds = 3

# Dict of minimum and maximum of y-axis for trajectory plots. Needs to be adjusted manually by visually checking plots.
y_min_dict = {976: 0.905, 980: 0.89, 1002: 0.5, 1018: 0.5, 1019: 0.965, 1021: 0.883, 1040: 0.955, 1053: 0.55, 1461: 0.6, 41160: 0.4}
y_max_dict = {976: 1.0, 980: 1.0, 1002: 0.84, 1018: 0.85, 1019: 1.0, 1021: 0.975, 1040: 0.995, 1053: 0.685, 1461: 0.86, 41160: 0.8}

