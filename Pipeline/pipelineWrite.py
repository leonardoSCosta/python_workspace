pipe_name = './python-pipe'

print("Writting to pipeline")
with open(pipe_name, 'w') as pipe_f:
    pipe_f.write("Teste 1 2 3")
    pipe_f.close()
print("Writting complete")
