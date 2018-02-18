# -*- coding: utf-8 -*-
"""
Copyright (C) 2018 Jacob Barhak
 
This file is part of the ClusterWebLog . ClusterWebLog is free software: you 
can redistribute it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

ClusterWebLog is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.


"""

# Cluster Web log analysis of the MSNBC data
# This data is available thanks to msnbc.com and was downloaded from
# http://archive.ics.uci.edu/ml/datasets/msnbc.com+anonymous+web+data
# The data is hosted by: Lichman, M. (2013). UCI Machine Learning Repository
# [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
# School of Information and Computer Science.

import numpy
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.resources import CDN
from bokeh.palettes import Viridis256
from bokeh.embed import file_html
from bokeh.transform import dodge
from sklearn.cluster import KMeans
import pandas

# load data
DataFile = open('msnbc990928.seq')
TextLines = DataFile.readlines()
LegendLine = TextLines[2]
# Read the code lines
Legend = ['Dummy'] + LegendLine.split()
# These are codes from 1 to N - exlude the dummy 0
LegendCodes = list(range(1,len(Legend)))

# Now read the data sequences
Sequences = [[int(Entry) for Entry in Line.split()] for Line in TextLines[7:]]
NumberOfRecords = len(Sequences)

# Now organize the data as counts
CountMatrix = numpy.zeros((NumberOfRecords,len(Legend)))

SequenceSets = [set(Sequence) for Sequence in Sequences]

PairMatches = numpy.zeros( (len(Legend), len(Legend)))
for Code1 in LegendCodes:
    for Code2 in LegendCodes:
        for (Sequence,SequenceSet) in zip(Sequences,SequenceSets):
            if Code1 in SequenceSet and Code2 in SequenceSet:
                if Code1 == Code2:
                    if Sequence.count(Code1)==1:
                        # Only single page in record
                        PairMatches[Code1][0] += 1
                        PairMatches[0][Code2] += 1
                    else:
                        # Page repeated 
                        PairMatches[Code1][Code2] += 1
                else:
                    # pages correlate
                    PairMatches[Code1][Code2] += 1
            
PairMatchesProportion = PairMatches/NumberOfRecords
MaxProportion = PairMatchesProportion.max() 
    
Source = ColumnDataSource(data=dict(
    Code1 = Legend*len(Legend) ,
    Code2 = list(numpy.repeat(Legend, len(Legend))),
    Color = [Viridis256[int((1-Entry/MaxProportion)**8*255)] for Entry in PairMatchesProportion.flatten()] , 
    Matches = PairMatches.flatten(),
    Proportion = PairMatchesProportion.flatten(),
    Text = [("%2i%%"%round(Entry*100)) for Entry in PairMatchesProportion.flatten()],
    ))

# Plot the data we see

MyHover1 = HoverTool(
    tooltips=[
        ( 'Page1', '@Code1{%s}'),
        ( 'Page2', '@Code2{%s}' ),
        ( 'Matches', '@Matches' ),  
        ( 'Proportion', '@Text{%s}' ),
     ],
    formatters={
        'Code1' : 'printf',   
        'Code2' : 'printf',   
        'Matches' : 'numeral',   
        'Text' : 'printf', 
     },
    point_policy="follow_mouse"            
)

Plot = figure(title="Pair Matches", tools=[MyHover1], x_range=Legend, y_range=Legend)
Plot.xaxis.major_label_orientation = "vertical"
Plot.rect('Code1', 'Code2', 0.9, 0.9, source=Source,
       color='Color', line_color=None,
       hover_line_color='black', hover_color='Color')

Handle = Plot.text(source = Source, x=dodge('Code1', -0.45, Plot.x_range), y=dodge('Code2', -0.35, Plot.y_range), text="Text", text_color='red')
Handle.glyph.text_font_size='10pt'

Html = file_html(Plot, CDN, 'my plot')
OutFile = open('PairMatches.html','w')
OutFile.write(Html)
OutFile.close()

################ Clustering ######################

# Now convert the sequences to another form where the number of page views
# is stored for each legend
HitVectors = numpy.zeros((NumberOfRecords, len(Legend)), dtype = numpy.int0)

for (SequenceEnum,Sequence) in enumerate(Sequences):
    for Code in LegendCodes:
        HitVectors[SequenceEnum][Code]  += Sequence.count(Code)


# now lets try to cluster those together to find the representative cases
# Easiest way is through k-means
NumberOfClusters = 20
ClusteringAlgorithm = KMeans(n_clusters=NumberOfClusters, random_state=1)
MappedClusters = ClusteringAlgorithm.fit(HitVectors)
Labels = list(MappedClusters.labels_)
ClusterSizes = [Labels.count(Code) for Code in range(NumberOfClusters)]
Distances = MappedClusters.transform(HitVectors)
BestDistanceToEachClusterCenter = Distances.argmin(axis=0)
ClosestRepresentativeSamplesToClusterCenter = Distances.argmin(axis=0)


BestDistance = Distances.min(axis=1)
DistanceMean = numpy.mean(BestDistance)
DistanceSTD = numpy.std(BestDistance)

DistancesPerCluster = [ numpy.extract(MappedClusters.labels_== ClusterEnum, BestDistance) for ClusterEnum in range(NumberOfClusters)]
MinDistancePerCluster = [min(Entry) for Entry in DistancesPerCluster]
MaxDistancePerCluster = [max(Entry) for Entry in DistancesPerCluster]
MeanDistancePerCluster = [ numpy.mean(Entry) for Entry in DistancesPerCluster]
STDDistancePerCluster = [ numpy.std(Entry) for Entry in DistancesPerCluster]

print ('Cluster Sizes are: ' +str(ClusterSizes))

print ("Cluster#:       Size        Min        Max       Mean        STD")
for ClusterEnum in range(NumberOfClusters):
    print ("%8i: %10i %10.1f %10.1f %10.1f %10.1f" % (ClusterEnum, ClusterSizes[ClusterEnum], MinDistancePerCluster[ClusterEnum], MaxDistancePerCluster[ClusterEnum], MeanDistancePerCluster[ClusterEnum], STDDistancePerCluster[ClusterEnum]))

# Define outliers as:
# 1. Clusters of size less than 100
# 2. Points with distance greater than 6 STD from mean distance

ClusterSizeThreshold = 100
ClustersToFilterOut = [ClusterEnum for ClusterEnum, ClusterSize in enumerate(ClusterSizes) if ClusterSize < ClusterSizeThreshold]

OutlierMask1 = numpy.in1d(MappedClusters.labels_, ClustersToFilterOut)
Outliers1 = numpy.extract(OutlierMask1, range(NumberOfRecords))

print ("Detected %i outliers in the following small clusters "%len(Outliers1) + str(ClustersToFilterOut) )

RemovalThrehold = 3 * DistanceSTD + DistanceMean
OutlierMask2 = BestDistance > (RemovalThrehold)
Outliers2 = numpy.extract(OutlierMask2, range(NumberOfRecords))

print ("Detected %i outliers with distance larger than %f "%(len(Outliers2),RemovalThrehold) )

def OutputSamples (FileName, RecordNumbers, SortByVector = None, FirstLine = None):
    "Output list to file" 
    OutList = []
    if FirstLine is not None:  
        OutList.append(FirstLine)
    Columns = ["Enum"]+Legend[1:]+['Distance','Record']  
    if SortByVector is None:
        SortedRecordNumbers = RecordNumbers
    else:
        SortedRecordNumbers = sorted(RecordNumbers, key = lambda RecordNumber:(SortByVector[RecordNumber]))
    for RecordNumber in SortedRecordNumbers:
        OutList.append([RecordNumber]+list(HitVectors[RecordNumber][1:])+[BestDistance[RecordNumber]] + [Sequences[RecordNumber]])
    DataFrame = pandas.DataFrame(OutList, columns = Columns)
    DataFrame.to_csv(FileName)

OutputSamples ('OutliersCluster.csv', Outliers1, BestDistance) 
OutputSamples ('OutliersDistance.csv', Outliers2, BestDistance)  

# Output Cluster Files 
for ClusterEnum in range(NumberOfClusters):
    if ClusterEnum not in ClustersToFilterOut:
        ThisClusterRecordMask = 1- (OutlierMask1+OutlierMask2 + (MappedClusters.labels_ != ClusterEnum))
        RemainingRecords =  numpy.extract(ThisClusterRecordMask, range(NumberOfRecords))
        FirstLine = ['Cluster']+list(MappedClusters.cluster_centers_[ClusterEnum])[1:] + [None] + [" Size=%i Min=%.1f Max=%.1f Mean=%.1f STD%.1f" % ( ClusterSizes[ClusterEnum], MinDistancePerCluster[ClusterEnum], MaxDistancePerCluster[ClusterEnum], MeanDistancePerCluster[ClusterEnum], STDDistancePerCluster[ClusterEnum])]
        OutputSamples('Cluster%i.csv'%(ClusterEnum),RemainingRecords,BestDistance,FirstLine)






 

