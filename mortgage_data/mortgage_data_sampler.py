import random

input_file = "./mortgage_data.csv"
output_file = "./sample_mortgage_data.csv"

f_input = open(input_file)

f_output = open(output_file, "w")

for line in f_input:
	rand = random.random()
	if rand < 0.015:
		f_output.write(line)
		print(rand)

f_input.close()
f_output.close()
