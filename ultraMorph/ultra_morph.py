#!/usr/bin/python3.8
"""
Calcula a semana do ultrassom morfológico para as pacientes
"""
from datetime import date, timedelta
import pandas as pd

EXAM_WEEK = 40
WEEK = 'Semanas'
EXAM = 'Data do exame'
FORMAT = '%d/%m/%Y - %A'


def read_data():
    """
    Lê o arquivo de entrada
    """
    data = pd.read_csv("./input.csv", sep=";")
    print("=== Dados de entrada ===\n", data, '\n')
    return data


def find_date(patient_data):
    """
    Calcula a semana do exame para cada paciente
    """
    today = date.today()
    print("=== Dia de hoje ===\n", today.strftime(FORMAT), '\n')

    for index in range(0, len(patient_data)):
        patient = patient_data.iloc[index]

        exam_delta = timedelta(weeks=EXAM_WEEK - int(patient[WEEK]))

        exam_date = ''
        if exam_delta.days <= 0:
            exam_date = 'Já deveria ter feito o exame'
        else:
            exam_date = (exam_delta + today).strftime(FORMAT)
        patient_data.loc[index, EXAM] = exam_date
    print("=== Data dos exames ===\n", patient_data, '\n')


if __name__ == "__main__":
    find_date(read_data())
