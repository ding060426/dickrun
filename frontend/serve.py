import http.server, socketserver, os
os.chdir(r"C:\Users\98068\Desktop\dickrun-new-meeting\frontend")
server = socketserver.ThreadingTCPServer(("", 3000), http.server.SimpleHTTPRequestHandler)
print("Frontend: http://localhost:3000")
server.serve_forever()
