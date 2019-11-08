import os

def fix_date(oldDate):
	year = int(oldDate[0:4])
	month = int(oldDate[4:5])
	if month == 0: # fix error in previous code
		month = 12
		year = year-1
		
	if month <= 9:
		newMonthStr = '0'+str(month)
	else:
		newMonthStr = str(month)
		
	newDate = str(year)+'-'+newMonthStr
	return newDate

dataDir = "./parallel year/data/"
fileList = os.listdir(dataDir)
print(fileList)
f_output = open("compiled_mortgage_output.csv","w")
for file in fileList:
	print(file)
	for line in open(dataDir+file):
		lineCols = line.split(',')
		oldDate = lineCols[5]
		newDate = fix_date(oldDate)
		outputStr = lineCols[0]+','+lineCols[1]+','+lineCols[2]+','+lineCols[3]+','+lineCols[4]+','+newDate+','+lineCols[6]
		f_output.write(outputStr)
f_output.close()

#0     ,1 ,2    ,3,  4,5,     6
#180000,PA,66849,6.3,0,199909,751
#amount,state,income,interest,delinq,date,credit