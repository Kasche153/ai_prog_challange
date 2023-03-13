import csv

file = open("output.csv", "w", newline='')
writer = csv.writer(file)

with open("TrainOnMe-6.csv", ) as f:
    reader = csv.reader(f, delimiter=",",)
    for i,  row in enumerate(reader):
        if(len(row) == 14):
            writer.writerow(row)
        

    
