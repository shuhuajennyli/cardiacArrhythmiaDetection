import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

#files = []

#adding all the wave and annotation files to a list in alphabetical order
#for dirname, _, filenames in os.walk('/home/cst2020/ScienceFair/mitbih-database/mitbih_database'):
    #for filename in filenames:
       # files.append(os.path.join(dirname, filename))
       # files = sorted(files)
        #print(files)

#making the function that will turn the files into what I want

#def process(wave, annotations):
for x in range(1):
	wave = pd.read_csv('/home/cst2020/ScienceFair/mitbih-database/mitbih_database/214.csv',header=None)
	annotations = pd.read_table('/home/cst2020/ScienceFair/mitbih-database/mitbih_database/214annotations.txt', delim_whitespace=True, header=None, quoting=3)
	
	print(annotations)


	#taking the second row of the annotations, which is the sampling number of the QRS complex peak of beats or 
	beat = annotations.iloc[1:,1]
	beat = beat.astype(int)

	print(beat)

	beat_type = annotations.iloc[1:,2]
	beat_type = beat_type.astype(str)
	print(beat_type)


	rhythm_type = annotations.iloc[1:,6]
	rhythm_type = rhythm_type.astype(str)
	print(rhythm_type)

	wave = wave.iloc[1:,1]
	wave = wave.astype(int)
	wave = wave.values.reshape(1,650000)
	wave = pd.DataFrame(wave)
	wave1 = wave.iloc[0,:650000]


	start = 0





	#start at one because the index of beat_type starts at 1 after iloc
	rhythm_change = []
	for x in range(1,len(beat_type)):
		if beat_type[x] == '+':
			if start == 0:
				start = 1
				start_type = rhythm_type[x]
				print("\n===============================================")
				print("Found (+), starting with ..." + start_type)
				print(x)
				print(start_type)
			elif start == 1:
				start = 0
				start_type = rhythm_type[x]
				print("Found (+), ending with..." + start_type)
				print(x)
				print("===============================================\n")
#				print(start_type)

		else:
			if start == 0:
				print("processing normal case...")
			else:
				print("processing + case - need to change it's type to " + start_type)

			print(x)
			print(rhythm_type[x])
#			print(start_type)

			#rhythm_change.append(x)
	#print(rhythm_change)






'''
	#processed = open("/home/cst2020/ScienceFair/processed2.csv", "a+")

	#adding 144 samplings to the front and back of the peak of the QRS complex assuming each beat is 0.8 seconds (80bpm)
	for x in range(len(realbeat)-1):
		if annotations.iloc[x, 0] == 'N' or annotations.iloc[x, 0] == 'Q' or annotations.iloc[x,0] == 'V' or annotations.iloc[x,0] == 'S' or annotations.iloc[x,0] == 'F':
			
			#making sure there is enough data to get the beat wave, or else, there will be an additional annotation
			if realbeat[x] > 144 and realbeat[x] < 649856: 
				segment_list = wave.iloc[0,realbeat[x]-144:realbeat[x]+144].tolist()

				if annotations.iloc[x, 0] == 'N':
					segment_type = '0'
				elif annotations.iloc[x, 0] == 'Q':
					segment_type = '1'
				elif annotations.iloc[x, 0] == 'V':
					segment_type = '2'
				elif annotations.iloc[x, 0] == 'S':
					segment_type = '3'
				elif annotations.iloc[x, 0] == 'F':
					segment_type = '4'


				#for item in segment_list:
					#processed.write("%s," % item)
		
				#processed.write(segment_type)
				#processed.write("\n")

	#processed.close()



print(len(files))



for x in range(96):
	if x % 2 == 0:
		process(files[x], files[x+1])
		print(files[x])


'''