#!/usr/bin/python3

import cgi 
import cgitb 
    
f = open("index.shtml", "r")
user_input = open("user_input", 'w')
page = f.read()
form = cgi.FieldStorage()

v = form.getvalue("guess")
if(v):
    user_input.write(v)
    user_input.close()
    print("Content-type: text/html\n\n")
    print(page)
    print("You typed:\n")
    print()
    print(v)
