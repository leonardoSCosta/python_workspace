# This creates a 10x10 matrix that contains the value 20
teste = [[20 for n in range(10)] for n in range(10)]
# Print the matrix
print("A 10x10 matrix: \n",teste)
# Access a single index
print("Matrix[0][0] = ",teste[0][0])

# The for variable can also be used like this:
teste = [[(n+1)/2 for n in range(10)] for n in range(10)]
print("A 10x10 matrix: \n",teste)
# Access a single index
print("Matrix[0][0] = ",teste[0][0])
