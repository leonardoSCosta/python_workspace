#!/usr/bin/env python3

dadosN = ["name = ","age = ","wants email = "]
dados = ["","",""]
test = input()
print(test)

for n in range(3):
	dados[n] = input(dadosN[n])
print("\n")

for n in range(3):
        print(dadosN[n]+dados[n])
