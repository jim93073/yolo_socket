from websocket import create_connection

def client_handle():
    ws = create_connection('ws://127.0.0.1:5000/echo')
    while ws.connected:        
#             ws.send('RECEIVED') 
        result = ws.recv()  
        print(result)            
            
        # ws.close()
    print("OUT")
if __name__ == "__main__":
    client_handle()
