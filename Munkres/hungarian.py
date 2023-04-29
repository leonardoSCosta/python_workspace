from munkres import Munkres, print_matrix

matrix = [[7100.48068795346, 7188.25889906589, 7336.42324297065, 7866.89169875879],
          [3577.67522282278, 4135.51689635044, 4722.60944817587, 5765.57031003872],
          [3058.95406961268, 3671.19871431662, 4300.697617829, 5370.64698150977],
          [2903.30845760488, 3542.55557472286, 4191.41980717752, 5268.60977867976]]

m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0

for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')

print(f'total cost: {total}')
