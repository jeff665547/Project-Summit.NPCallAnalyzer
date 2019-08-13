# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:04:01 2019

@author: Chao-Hsi Lee
"""

import tkinter as tk
 
def callBackFunc():
    print("Oh. I'm clicked")
    
app = tk.Tk() 
app.geometry('150x100')

chkValue = tk.BooleanVar() 
# chkValue.set(True)
 
chkExample = tk.Checkbutton(app, text='Check Box', 
                            var=chkValue) 
chkExample.pack()

app.mainloop()