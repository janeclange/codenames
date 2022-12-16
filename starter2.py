#!/usr/bin/python3

import cgi 
import cgitb 
import subprocess

cgitb.enable()
print("Content-type: text/html\n\n")
print("Made it here <br/>")

f = open("index.shtml", "r")
page = f.read()
print(page)

proc = subprocess.Popen(["python3 codenames3.py"], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try: 
    outs, errs = proc.communicate()
    #outs, errs = proc.communicate(input=bytes("guess\n", encoding="utf-8"), timeout=15)
except TimeoutExpired: 
    proc.kill()
    outs, errs = proc.communicate()
print(outs.decode())
