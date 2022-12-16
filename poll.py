#!/usr/bin/python3

import cgi 
import cgitb

print("Content-type: text/html\n\n")
f = open("index.shtml", "r")
page = f.read()
print(page)
f = open("current_output", "r")
out = f.read()
print(out)


