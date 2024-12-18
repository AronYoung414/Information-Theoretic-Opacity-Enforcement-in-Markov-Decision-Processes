import matplotlib.pyplot as plt
import pickle

ex_num = 3
iter_num = 3000

iteration_list = range(iter_num)

with open(f'../Data_Initial/entropy_values_{ex_num}.pkl', 'rb') as file:
    entropy_list = pickle.load(file)

with open(f'../Data_Initial/value_function_list_{ex_num}', 'rb') as file:
    threshold_list = pickle.load(file)


figure, axis = plt.subplots(2, 1)

axis[0].plot(iteration_list[:2000], entropy_list[:2000], label='Entropy')
axis[1].plot(iteration_list[:2000], threshold_list[:2000], label='Estimated Cost')
plt.xlabel("Iteration number")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.savefig(f'../Data_Initial/graph_{ex_num}.png')
plt.show()

with open(f'../Data_Initial/final_control_policy_{ex_num}.pkl', 'rb') as file:
    control_policy = pickle.load(file)

print(control_policy)