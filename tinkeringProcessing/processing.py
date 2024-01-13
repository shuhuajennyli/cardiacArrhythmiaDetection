import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

files = []

#adding all the wave and annotation files to a list in alphabetical order
for dirname, _, filenames in os.walk('/home/cst2020/ScienceFair/mitbih-database/mitbih_database'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        files = sorted(files)
        print(files)

#making the function that will turn the files into what I want
def process(wave, annotations):
	wave = pd.read_csv(wave,header=None)
	annotations = pd.read_table(annotations, delim_whitespace=True, header=None, quoting=3)
	
	#taking the second row of the annotations, which is the sampling number of the QRS complex peak of beats or 
	beat = annotations.iloc[2:,1]
	beat = beat.astype(int)

	annotations = annotations.iloc[2:,2]
	annotations = annotations.astype(str)
	annotations = pd.DataFrame(annotations)

	wave = wave.iloc[1:,1]
	wave = wave.astype(int)
	wave = wave.values.reshape(1,650000)
	wave = pd.DataFrame(wave)
	wave1 = wave.iloc[0,:650000]

	max, _ = find_peaks(wave1, height = 1100, distance = 80)


	realbeat = []
	for x in max:
		for y in beat:
			if abs(x-y) == 0 or abs(x-y) == 1:
				realbeat.append(y)


	processed = open("/home/cst2020/ScienceFair/processed.csv", "a+")
	#processed.write('processing wave' + '\n')
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


				for item in segment_list:
					processed.write("%s," % item)
		
				processed.write(segment_type)
				processed.write("\n")

	#processed.write('processomg end' + '\n')
	processed.close()


#total 96 files including csv and annotations
print(len(files))

#plugging in the files in the function
#for x in range(len(files)):
#214.csv does not work (68 for debugging)
for x in range(96):
	if x % 2 == 0:
		process(files[x], files[x+1])
		print(files[x])




'''
#TESTING FIRST BEAT
print("\n=>TESTING FIRST BEAT (start to end...")
start = realbeat[88]-144
end = realbeat[88]+144

print("\n=>TESTING wave...")
print(wave)
print(type(wave))

print("\n=>TESTING random...")
random = wave.iloc[0,start:end]
#random = wave.iloc[0,22970:23258]
print(type(random))

segment_list=wave.iloc[0,start:end].tolist()
print(segment_list)
print(len(segment_list))
'''
