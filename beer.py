#!/usr/bin/python3.8
import sys


def calcPreco(tam, val) -> float:
    aux = 1e3/tam
    return (aux * val)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python beer.py NomeCerveja Valor[R$] Tamanho[ml]")
        print("Informe todos os argumentos!")
    else:
        nome = sys.argv[1]
        val = float(sys.argv[2])
        tam = int(sys.argv[3])
        print("%s - %d ml | Preço -> %.2f R$/L" %
              (nome, tam, calcPreco(tam, val)))
