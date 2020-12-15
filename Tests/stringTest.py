long_str = "Leonardo da Silva Costa"

# inverts one string using the -1 operator
inverse = long_str[::-1]
# get only the even index values
even_str = long_str[::2]
# a substring
subs_str = long_str[3:16]
print("String: ", long_str)
print("Inverse of the string: ",inverse)
print("Even indexes of the string: ",even_str)
print("Substring: ", subs_str)

# using for loop

for ltr in long_str[::-1]:
	print(ltr)
