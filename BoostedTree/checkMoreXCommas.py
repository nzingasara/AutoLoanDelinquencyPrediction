fr = open("mortgage_data.csv", "r")
for i, line in enumerate(fr):
    if line.count(",") > 6:
        print(i+1)
fr.close()
