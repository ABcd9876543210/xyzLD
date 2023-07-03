# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:58:56 2023

@author: Aditi
"""

from __future__ import with_statement
import re 
words =[]
testwords=[]
ans = []

print('MENU')
print('---------')
print('1 :Hash tag segmentation')
print('2 :URL segmentation')
print("ener the input choisce for performing word segmentation")

choice = int(input())
if choice == 1:
    text = "#whatismyname"
    print('input with hashtag', text)
    pattern = re.compile("[^\w']")
    a = pattern.sub('',text)
elif choice == 2:
    text = "www.whatismyname.com"
    print("inout with URL",text)
    a= re.split('\s|(?!\d)[,.](?!\d)', text)
    splitwords =["www","com","in"]
    a="".join([each for each in a if each not in splitwords])
else:
    print("wrong choice.. tray again")
print(a)

for each in a:
    testwords.append(each)
test_length = len(testwords)
with open('D:/abhishek/model college/sem4/NLP/word.txt','r') as f:
    lines = f.readlines()
    words = [(e.strip()) for e in lines]
def seg(a,length):
    ans=[]
    for k in range(0, length+1):
        if a[0:k] in words:
            print(a[0:k],'-appears in the corpus')
            ans.append(a[0:k])
            break
        if ans!= []:
            g = max(ans, key=len)
            return g
test_tot_itr =0
answer =[]
score=[]
N = 37
M=0
C=0
while test_tot_itr <test_length:
    answer_words = seg(a, test_length)
    if answer_words != 0:
        test_itr = len(answer_words)
        answer.append(answer_words)
        a = a[test_itr:test_length]
        test_tot_itr += test_itr
aft_seg = "".join([each for each in answer])
print("output")
print("----")
print(aft_seg)
c =len(answer)
print("score:", score)
                    