pipe_name = './python-pipe'

print("Reading from pipeline")
#  while(1):
with open(pipe_name, 'r') as pipe_f:
    if(pipe_f.readable()):
        data = pipe_f.readline()
        print(data)
        pipe_f.close()
print("Reading complete")
