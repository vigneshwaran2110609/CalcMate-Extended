import csv
from dataset import textbook_examples

print(len(textbook_examples))

with open('dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["problem", "reasoning"])  

    for example in textbook_examples:
        writer.writerow([example["problem"], example["reasoning"]])

print("CSV file 'textbook_problems.csv' has been created with the problems and reasoning.")
