#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:49:22 2021

@author: ethanayari
"""

#%%
def create_fracs(percent_arr, rocks):
    percentarray_s = percent_arr
    
    #Convert mineral abundances to to fractions 
    rocks['abundance1'] =  rocks['abundance1'].astype(float)/100
    rocks['abundance2'] =  rocks['abundance2'].astype(float)/100
    rocks['abundance3'] =  rocks['abundance3'].astype(float)/100
    rocks['abundance4'] =  rocks['abundance4'].astype(float)/100
    rocks['abundance5'] =  rocks['abundance5'].astype(float)/100
    rocks['abundance6'] =  rocks['abundance6'].astype(float)/100
    rocks['abundance7'] =  rocks['abundance7'].astype(float)/100
    rocks['abundance8'] =  rocks['abundance8'].astype(float)/100
    frac_array = percentarray_s/100.0
    
    return frac_array


              
              
    