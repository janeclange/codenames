#!/usr/bin/python3

import cgi 
import cgitb 
import waitforinput
import os


r, w = os.pipe()
pid = os.fork()

if pid: #parent process
#    print("Content-type: text/html\n\n")
    f = open("index.shtml", "r")
    page = f.read()
    #print(page)
    #print("Started!")
    os.close(w)
    r = os.fdopen(r)
    print("Process ID:", os.getpid())
    print("Parent reading")
    str = r.read()
    print("Parent reads =", str)
else: #child process 
    os.close(r)
    w = os.fdopen(w, 'w')
    print("Process ID:", os.getpid())
    print("Parent's process ID:", os.getppid())
    print("Child writing")
    writes = "testing"
    w.write(writes)
    print("Child writes =", writes)
    w.close()


