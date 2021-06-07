import websocket

websocket.enableTrace(True)
try:
    ws = websocket.create_connection("ws://localhost:8081/api/control")
except ConnectionRefusedError:
    print('Connection refused')
    exit(1)
except Exception as expt:
    print(expt, type(expt))
    exit(1)

ws.send('{"change":{"newCommand":{"command":{"type":"HALT","forTeam":"UNKNOWN"}},"origin":"UI"}}')

#  result = ws.recv()
#  print("Received '%s'" % result)
#  result = ws.recv()
#  print("Received '%s'" % result)
