#!/usr/bin/python3

import cgi 
import cgitb 

form = cgi.FieldStorage()
v = form.getvalue("guess")
print("Content-type: text/html\n\n")
print("success\n")
print(v)
