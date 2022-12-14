#!/usr/bin/python3

import cgi 
import cgitb 
f = open("index.shtml", "r")
page = f.read()

form = cgi.FieldStorage()
v = form.getvalue("guess")
print("Content-type: text/html\n\n")
print(page)
print("success\n")
print(v)
