import os
import pandas as pd
import numpy as np


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
	
	print(annotations)


	#taking the second row of the annotations, which is the sampling number of the QRS complex peak of beats or 
	beat = annotations.iloc[1:,1]
	beat = beat.astype(int)

	print(beat)

	#taking the third row of the annotations, which is the beat type
	beat_type = annotations.iloc[1:,2]
	beat_type = beat_type.astype(str)
	print(beat_type)

	#taking the seventh row of the annotations, which is the rhythm type
	rhythm_type = annotations.iloc[1:,6]
	rhythm_type = rhythm_type.astype(str)
	print(rhythm_type)

	#the entire wave (samplings and its respective y value on graph)
	wave = wave.iloc[1:,1]
	wave = wave.astype(int)
	wave = wave.values.reshape(1,650000)
	wave = pd.DataFrame(wave)
	wave1 = wave.iloc[0,:650000]

	#finding out when there is a rhythm change and adding to a list
	rhythm_change = []
	for x in range(1, len(beat)+1):
		if beat_type[x]== '+':
			rhythm_change.append(x)

	print(rhythm_change)

	#a list of rhythm annotations, there is an 'extra' so the index of the annotations matches the index of the beat
	rhythmannotations = ['extra']
	
	for x in range(1, len(beat)+1):
		if rhythm_type[x] == 'nan':
			#finds the closest rhythm change sampling number
			closest = min(rhythm_change, key = lambda y:abs(y-x))
			#finds the closest rhythm change sampling number before the beat
			if closest > x:
				closest = rhythm_change[rhythm_change.index(closest) -1 ]
			rhythmannotations.append(rhythm_type[closest])
		elif rhythm_type[x] != 'nan':
			rhythmannotations.append(rhythm_type[x])

	print(rhythmannotations)

	#creating a file of data
	processed = open("/home/cst2020/ScienceFair/processedfinal.csv", "a+")


	#adding 144 samplings to the front and back of the peak of the QRS complex assuming each beat is 0.8 seconds (80bpm)
	for x in range(1, len(beat)+1):
		if beat_type[x] == 'N' or beat_type[x] == 'Q' or beat_type[x] == 'x' or beat_type[x] == 'f' or beat_type[x] == '/' or beat_type[x] == '!' or beat_type[x] == 'E' or beat_type[x] == 'V' or beat_type[x] == 'F' or beat_type[x] == 'J' or beat_type[x] == 'S' or beat_type[x] == 'a' or beat_type[x] == 'L' or beat_type[x] == 'R' or beat_type[x] == 'A':
			
			#making sure there is enough data to get the beat wave, or else there will be an additional annotation
			if beat[x] > 144 and beat[x] < 649856: 
				segment_list = wave.iloc[0,beat[x]-144:beat[x]+144].tolist()

				if rhythmannotations[x] == '(AB':
					segment_type = '0'
				elif rhythmannotations[x] == '(AFIB':
					segment_type = '1'
				elif rhythmannotations[x] == '(AFL':
					segment_type = '2'
				elif rhythmannotations[x] == '(B':
					segment_type = '3'
				elif rhythmannotations[x] == '(BII':
					segment_type = '4'
				elif rhythmannotations[x] == '(IVR':
					segment_type = '5'
				elif rhythmannotations[x] == '(N':
					segment_type = '6'
				elif rhythmannotations[x] == '(NOD':
					segment_type = '7'
				elif rhythmannotations[x] == '(P':
					segment_type = '8'
				elif rhythmannotations[x] == '(PREX':
					segment_type = '9'
				elif rhythmannotations[x] == '(SBR':
					segment_type = '10'
				elif rhythmannotations[x] == '(SVTA':
					segment_type = '11'
				elif rhythmannotations[x] == '(T':
					segment_type = '12'
				elif rhythmannotations[x] == '(VFL':
					segment_type = '13'
				elif rhythmannotations[x] == '(VT':
					segment_type = '14'

				for item in segment_list:
					processed.write("%s," % item)
				
				processed.write(segment_type)
				processed.write("\n")

	processed.close()

#carrying out the process function on all data
for x in range(96):
	if x % 2 == 0:
		process(files[x], files[x+1])
		print(files[x])