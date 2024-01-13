import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks





wave = pd.read_csv('/home/cst2020/ScienceFair/mitbih-database/105.csv',header=None)
annotations = pd.read_table('/home/cst2020/ScienceFair/mitbih-database/105annotations.txt', delim_whitespace=True, header=None)

beat = annotations.iloc[2:,1]
beat = beat.astype(int)
print(beat)


annotations = annotations.iloc[2:,2]
annotations = annotations.astype(str)
annotations = pd.DataFrame(annotations)

wave = wave.iloc[1:,1]
wave = wave.astype(int)
wave = wave.values.reshape(1,650000)
wave = pd.DataFrame(wave)
wave1 = wave.iloc[0,:650000]
#print(wave)

max, _ = find_peaks(wave1, height = 1100, distance = 80)
#max = pd.DataFrame(max)
#print(max)

#plt.plot(wave1)
#plt.plot(max, wave[max], "x")
#plt.show()

#max=max[0].tolist()
#guessbeat = beat[0].tolist()

realbeat = []
for x in max:
	for y in beat:
		if abs(x-y) == 0 or abs(x-y) == 1:
			realbeat.append(y)

#print(realbeat)
print("hi")
print(annotations.iloc[2687, 0])


dataset = []

mydatasetfile = open("/home/cst2020/ScienceFair/mydatasetfile.csv", "w")

#adding 144 samplings to the front and back of the peak of the QRS complex assuming each beat is 0.8 seconds (80bpm)
for x in range(len(realbeat)-1):
	if annotations.iloc[x, 0] == 'N' or annotations.iloc[x, 0] == 'Q' or annotations.iloc[x,0] == 'V' or annotations.iloc[x,0] == 'S' or annotations.iloc[x,0] == 'F':
		segment_list = wave.iloc[0,realbeat[x]-144:realbeat[x]+144].tolist()
		print(len(segment_list))

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

		print(segment_type)

		for item in segment_list:
			mydatasetfile.write("%s," % item)
		
		mydatasetfile.write(segment_type)
		mydatasetfile.write("\n")

#		dataset.append(wave.iloc[0,realbeat[x]-144:realbeat[x]+144])
#		dataset.append(wave.iloc[0,realbeat[x]-144:realbeat[x]+144])
#		dataset.append(annotations.iloc[x, 0])
		
mydatasetfile.close()


#print(dataset)







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






#np.savetxt(r'/home/cst2020/ScienceFair/calculatedbeat_distance80.txt', max.values, fmt='%d')
#np.savetxt(r'/home/cst2020/ScienceFair/datasetbeat_distance80.txt', annotations.values, fmt='%d')