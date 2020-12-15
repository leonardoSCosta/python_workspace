import pandas as pd

df = pd.read_excel("Papers.xls")

#df_out = df[['First Name','Last Name','E-mail']]
#df_out.to_csv("reviewers.csv")

assignFile = open('AssignmentsTemplate.xml','r')

assignTemp = assignFile.read()

assignSTR = ""

#print(assignTemp)
#print(df['Paper ID'])


for paperID in df['Paper ID']:
	listAux = list(assignTemp)
	listAux.insert(listAux.index("\"")+1,str(paperID))
	assignSTR += "".join(listAux) + "\n"
	print(assignSTR)


assignFile.close()
assignFile = open('AssignMod.xml','w')
assignFile.write(assignSTR)
assignFile.close()
