#!/usr/bin/env python

__author__ = 'N.H.'

def genchrnum(chrom):
  """ Sort by chromosome """
  if chrom:
    num = chrom[3:]
    if   num == 'X': num = 23
    elif num == 'Y': num = 24
    elif num == 'M': num = 25
    else: num = int(num)
  else:
    num = 0
  return num

