import matplotlib.pyplot as plt
import running_example_no_masking
import running_example

# Run experiments

# Experiment with no masking.
# running_example_no_masking.run_experiment_with_no_masking(iter_num=1000, batch_size=100, V=1500, T=3, exp_number=1)
# # Experiment with masking for threshold = 60. --> toggle prior_compute_flag=1 to compute the prior conditional entropy.
running_example.running_example(iter_num=500, batch_size=100, V=300, T=3, threshold=60, prior_compute_flag=0, exp_number=2)
# # Experiment with masking threshold = 20
# running_example.running_example(iter_num=1000, batch_size=100, V=1500, T=3, threshold=20, prior_compute_flag=0, exp_number=3)


print("End of illustrative examples.")
