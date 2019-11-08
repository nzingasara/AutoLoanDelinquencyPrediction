orig_file = "../data/historical_data1_1999/historical_data1_Q11999.txt"
month_file = "../data/historical_data1_1999/historical_data1_time_Q11999.txt"

def calculateMortgagePayment(p, r, n):
	return p*(((r/12)*(1+(r/12))**n)/((1+(r/12))**n-1))

def calculateIncome(dti, p, r, n):
	if (dti == 9.99):
		return "NaN"
	else:
		return 12*calculateMortgagePayment(p,r/100,n)/dti



def runFile(orig_file, month_file):
	f_orig = open(orig_file)
	f_month = open(month_file)
	searchIndex = 0
	currIndex = 0
	
	for line in f_orig:
		#print(line)
		cols = line.split('|')
		
		loanId = cols[19] # ORIG col 20, MONTH col 1
		loanAmount = int(cols[10]) # ORIG UPB col 11
		usState = cols[16] # ORIG col 17
		interestRate = float(cols[12]) # ORIG col 13
		disbursalDate = str(int(cols[1])-1) # ORIG col 2 (minus one month)
		creditScore = cols[0] # ORIG col 1
		
		dti = float(cols[9])/100
		n = int(cols[21])
		income = calculateIncome(dti, loanAmount, interestRate, n) # ORIG debt/income col 10 (monthly debt/monthly income)

		delinquent = "NaN"
		for line2 in f_month:
			currIndex += 1
			cols2 = line2.split('|')
			if (cols2[0] == loanId):
				if (cols2[3] != "R"):
					if (int(cols2[3]) >= 3):
						delinquent = "1" # MONTH col 4, TRUE if val > 3
					else:
						delinquent = "0"
			elif (delinquent != "NaN"):
				searchIndex += currIndex
				currIndex = 0
				break
		if (delinquent == "NaN"): # loanId not found in month data (or is REO, which is uncommon)
			f_month = open(month_file) # start file from beginning
			for _ in range(searchIndex): # skip to previously found loanId
				next(f_month)
			currIndex = 0
				
		if (income == "NaN"):
			incomeString = "NaN"
		else:
			incomeString = str(int(round(income)))
		outputFile.write(str(loanAmount)+","+usState+","+incomeString+","+str(interestRate)+","+str(delinquent)+","+disbursalDate+","+creditScore+"\n")

	
'''	
outputFile = open("mortgage_data.csv","w")
for year in range(1999,2018):
	for quarter in range(1,5):
		print(str(year)+","+str(quarter))
		orig_filename = "../data/historical_data1_"+str(year)+"/historical_data1_Q"+str(quarter)+str(year)+".txt"
		month_filename = "../data/historical_data1_"+str(year)+"/historical_data1_time_Q"+str(quarter)+str(year)+".txt"
		runFile(orig_filename, month_filename)

'''
outputFile = open("test.csv","w")
orig_filename = "../data/test_orig.txt"
month_filename = "../data/test_month.txt"
runFile(orig_filename, month_filename)

outputFile.close()
'''
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

loanID = # ORIG col 20, MONTH col 1
reportingMonth = # MONTH col 2
numTerms = # ORIG col 22

F1YYQnXXXXXX •F1 = product (Fixed Rate Mortgage);  •YYQn = origination year and quarter; and,  •XXXXXX = randomly assigned digits
'''