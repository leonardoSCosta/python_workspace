fileNome = open("nomes.txt", 'r')
nomes = fileNome.read()
print("Nomes fora de formatação: " + nomes)

autores = nomes.split(',')
nomeOrdenado = ""

for n,autor in enumerate(autores):
    nomeSobrenome = autor.strip().split(" ")
    aux = " ".join(nomeSobrenome[:len(nomeSobrenome)-1])
    aux = nomeSobrenome[-1] + ", " + aux
    if n < len(autores)-1:
        nomeOrdenado += aux + " and "
    else:
        nomeOrdenado += aux

print("Nomes formatados: " +nomeOrdenado)