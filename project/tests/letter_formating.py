import numpy as np
import cv2 as cv
import os
from string import ascii_lowercase
from string import ascii_uppercase




files = os.listdir('letter_pictures')
letters = []

for file in files:
	if file.endswith('.PNG'):
		
		letters.append(cv.imread('letter_pictures\\'+file))


small = []
big = []

for let in ascii_lowercase:
	small.append(let)


for lete in ascii_uppercase:
	big.append(lete)

even = 0

counter_small = 0
counter_big = 0

umlauts = ['Ae','ae','Oe','oe','Ue','ue']

umlaut=0

for letter in letters:
	letter= cv.resize(letter,(200,200))
		

	if umlaut<len(umlauts)+52 and umlaut >=52:

		if even % 2==1:
			cv.imwrite('letters_resized\\{}_small_resized'.format(str(umlauts[umlaut-52]))+'.png',letter)
			
			

		elif even % 2==0:
			cv.imwrite('letters_resized\\{}_big_resized'.format(str(umlauts[umlaut-52]))+'.png',letter)
			
		
	else:

		if even % 2==1 and counter_small<26:
			cv.imwrite('letters_resized\\{}_small_resized'.format(str(small[counter_small]))+'.png',letter)
			counter_small+=1
		elif even % 2==0 and counter_big<26:
			cv.imwrite('letters_resized\\{}_big_resized'.format(str(big[counter_big]))+'.png',letter)
			counter_big+=1
			

	even+=1
	umlaut+=1





