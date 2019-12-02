def calculateMortgagePayment(p, r, n):
	return p*(((r/12)*(1+(r/12))**n)/((1+(r/12))**n-1))

def calculateIncome(dti, p, r, n):
	if (dti == 9.99):
		return "NaN"
	else:
		return 12*calculateMortgagePayment(p,r/100,n)/dti


def runFile(orig_file, month_file):
	f_month = open(month_file)
	f_orig = open(orig_file)
	f_orig_lines = f_orig.readlines()
	
	loanId = ""
	month_cols = []
	
	counter = 0
	setIndex = 0
	
	for month_line in f_month:
		#print(month_line)
		counter += 1
		if (counter%10000==0):
			print(str(counter)+", "+str(year))
		next_month_cols = month_line.split('|')
		nextLoanId = next_month_cols[0]
		
		if (loanId != "" and nextLoanId != loanId): # process previous line
			if (month_cols[3] != "R"):
				if (int(month_cols[3]) >= 3):
					delinquent = "1" # MONTH col 4, TRUE if val > 3
				else:
					delinquent = "0"
			else:
				delinquent = "NaN"
			
			#for orig_line in f_orig:
			orig_index = 0
			for orig_line in f_orig_lines[setIndex:]:
				orig_index += 1
				orig_cols = orig_line.split('|')
				if (orig_cols[19] == loanId):
					loanAmount = int(orig_cols[10]) # ORIG UPB col 11
					usState = orig_cols[16] # ORIG col 17
					interestRate = float(orig_cols[12]) # ORIG col 13
					disbursalDate = str(int(orig_cols[1])-1) # ORIG col 2 (minus one month)
					creditScore = orig_cols[0] # ORIG col 1
					
					dti = float(orig_cols[9])/100
					n = int(orig_cols[21])
					income = calculateIncome(dti, loanAmount, interestRate, n) # ORIG debt/income col 10 (monthly debt/monthly income)
							
					if (income == "NaN"):
						incomeString = "NaN"
					else:
						incomeString = str(int(round(income)))
				
					outputFile.write(str(loanAmount)+","+usState+","+incomeString+","+str(interestRate)+","+str(delinquent)+","+disbursalDate+","+creditScore+"\n")
					setIndex += orig_index
					break
		
		month_cols = next_month_cols
		loanId = nextLoanId
					

outputFile = open("mortgage_data.csv","w")
year = 2006
for quarter in range(1,5):
	print(str(year)+","+str(quarter))
	orig_filename = "../../data/historical_data1_"+str(year)+"/historical_data1_Q"+str(quarter)+str(year)+".txt"
	month_filename = "../../data/historical_data1_"+str(year)+"/historical_data1_time_Q"+str(quarter)+str(year)+".txt"
	runFile(orig_filename, month_filename)

'''
outputFile = open("test.csv","w")
orig_filename = "../data/test_orig.txt"
month_filename = "../data/test_month.txt"
runFile(orig_filename, month_filename)
'''
outputFile.close()
'''
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

loanId = # ORIG col 20, MONTH col 1
reportingMonth = # MONTH col 2
numTerms = # ORIG col 22

F1YYQnXXXXXX •F1 = product (Fixed Rate Mortgage);  •YYQn = origination year and quarter; and,  •XXXXXX = randomly assigned digits
'''