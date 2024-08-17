import os
import numpy as np
import pandas as pd 
import seaborn as sns
from radar_scenes.sequence import Sequence
from radar_scenes.labels import Label, ClassificationLabel
from collections import Counter
from sklearn.neighbors import NearestNeighbors

import matplotlib as mpl
import matplotlib.pyplot as plt
    

class TrackInstanceData:
    def __init__(self):
        self.meanDist = 0
        self.stdDist = 0
        self.meanRCS = 0
        self.stdRCS = 0 
        self.numPts = 0
        self.labelID = 0

    def set_data(self, track_id, radar_data):
        idx = np.where(radar_data["track_id"] == track_id)[0]
        self.radar_data_inst = radar_data[idx] 

        self.meanDist = np.mean(self.radar_data_inst["range_sc"])
        self.stdDist = np.std(self.radar_data_inst["range_sc"])

        self.meanRCS = np.mean(self.radar_data_inst["rcs"])
        self.stdRCS = np.std(self.radar_data_inst["rcs"])

        self.meanVel = np.mean(self.radar_data_inst["vr_compensated"])
        self.stdVel =  np.std(self.radar_data_inst["vr_compensated"])

        self.numPts = self.radar_data_inst.size
        self.labelID = self.radar_data_inst["label_id"][0]

    
class TrackStats:
    def __init__(self):
        self.meanDist = []
        self.stdDist = []
        self.meanRCS = []
        self.stdRCS = []
        self.meanVel = []
        self.stdVel = []
        self.LabelIDs = []
        self.numPtsPerTrack = []

    def add_track_data(self, trackData: TrackInstanceData) :
        self.meanRCS.append(trackData.meanRCS)
        self.stdRCS.append(trackData.stdRCS)
        self.meanDist.append(trackData.meanDist)
        self.stdDist.append(trackData.stdDist)
        self.meanVel.append(trackData.meanVel)
        self.stdVel.append(trackData.stdVel)
        self.LabelIDs.append(trackData.labelID)
        self.numPtsPerTrack.append(trackData.numPts)
        

    def get_RCS_data_per_label_id(self, label_id):
        idx = np.where(np.asarray(self.LabelIDs) == label_id)[0]
    
        return np.array(self.meanRCS)[idx],np.array(self.stdRCS)[idx], np.array(self.meanDist)[idx]

    def get_NumPts_data_per_label_id(self, label_id):
        idx = np.where(np.asarray(self.LabelIDs) == label_id)[0]
    
        return np.array(self.numPtsPerTrack)[idx], np.array(self.meanDist)[idx]


def count_label_ids(sequence: Sequence):
    """
    Iterate over all scenes in a sequence and collect for each scene, how many detections belong to each class.
    :param sequence: a measurement sequence
    :return: A list of dictionaries. Each dict contains as key the label ids and as value the number of times this
    label_id occured.
    """
    # iterate over all scenes in the sequence and collect the number of labeled detections per class
    labels_per_scene = []
    for scene in sequence.scenes():
        label_ids = scene.radar_data["label_id"]
        c = Counter(label_ids)
        labels_per_scene.append(dict(c.items()))
    return labels_per_scene

def get_radar_stats_per_track(trackstat: TrackStats, sequence: Sequence):

    for scene in sequence.scenes():
        track_ids = scene.radar_data["track_id"]
        label_ids = scene.radar_data["label_id"]

        valid_idx = np.where(label_ids != Label.STATIC.value)[0]
        unique_tracks = set(track_ids[valid_idx])
        unique_ids = np.array(list(unique_tracks))

        trackInst = TrackInstanceData()
        for track_id in unique_ids:
            trackInst.set_data(track_id,scene.radar_data)
            trackstat.add_track_data(trackInst)

    return trackstat

def count_unique_objects(sequence: Sequence):
    """
    For each scene in the sequence, count how many different objects exist.
    Objects labeled as "clutter" are excluded from the counting, as well as the static detections.
    :param sequence: A measurement sequence
    :return: a list holding the number of unique objects for each scene
    """
    objects_per_scene = []
 
    for scene in sequence.scenes():
        track_ids = scene.radar_data["track_id"]
        label_ids = scene.radar_data["label_id"]

        valid_idx = np.where(label_ids != Label.STATIC.value)[0]
        unique_tracks = set(track_ids[valid_idx])
        unique_ids = np.array(list(unique_tracks))

        objects_per_scene.append(len(unique_tracks))
    return objects_per_scene


def NearestPoint(target) :
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(target)
    distances, indices = nbrs.kneighbors(target)
    return distances, indices

    

def main():
    # MODIFY THIS LINE AND INSERT PATH WHERE YOU STORED THE RADARSCENES DATASET
    path_to_dataset = "C:/Users/eric_/Dropbox/Schooling/Eric/Technion/Project1/RadarScenes"

    # Define the *.json file from which data should be loaded
    filename = os.path.join(path_to_dataset, "data", "sequence_137", "scenes.json")

    if not os.path.exists(filename):
        print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        return

    # create sequence object from json file
    sequence = Sequence.from_json(filename)

    trackstats = pd.DataFrame(sequence.radar_data)
    staticObjs = trackstats[trackstats['label_id'] == 11]
    dynamicObjs = trackstats[trackstats['label_id'] != 11]
    print('The mean velocity of static objects is', np.mean(staticObjs['vr_compensated']), 'and std of', np.std(staticObjs['vr_compensated']))
    print('The mean velocity of dynamic objects is', np.mean(dynamicObjs['vr_compensated']), 'and std of', np.std(dynamicObjs['vr_compensated']))

    fig, axes = plt.subplots(1,2)

    sns.violinplot(staticObjs['vr_compensated'],ax = axes[0])
    sns.violinplot(dynamicObjs['vr_compensated'],ax = axes[1])

    plt.show()

    trackstats = trackstats.drop(trackstats[trackstats.label_id == 11].index)
    trackstats = trackstats.drop(trackstats[trackstats.label_id == 10].index)
    #trackstats = trackstats.filter('label_id'==0,axis=0)
    groupedStats = trackstats.groupby(['timestamp','track_id','label_id','sensor_id']).agg({'range_sc':['mean','std'],'rcs':['mean','std'],'vr_compensated':['mean','std']})
    groupedStats['count'] = trackstats.groupby(['timestamp','track_id','label_id','sensor_id']).size()
    groupedStats.reset_index()

    numSensors = trackstats['sensor_id'].max()
    """
    #checking for dependence on RCS and num of points    
    fig, axes = plt.subplots(numSensors, 2, figsize=(15, 5), sharey=True)

    for i in range(1,numSensors+1) :
        sns.scatterplot(data=groupedStats.reset_index()[groupedStats.reset_index()['sensor_id'] == i], x = ('range_sc','mean'), y=('rcs','mean'), hue=('label_id'),ax = axes[i-1,0])
    
#        sns.regplot(data=groupedStats.reset_index()[groupedStats.reset_index()['sensor_id'] == i], x = ('range_sc','mean'), y=('count'), hue=('label_id'),order=0.5,ax = axes[i-1,1])
        sns.scatterplot(data=groupedStats.reset_index()[groupedStats.reset_index()['sensor_id'] == i], x = ('range_sc','mean'), y=('count'), hue=('label_id'),ax = axes[i-1,1])


    plt.show()
    """

    groupNN = trackstats.drop(['sensor_id','range_sc','azimuth_sc','rcs','vr','vr_compensated','x_seq','y_seq','uuid','label_id'],axis=1)
    nps = groupNN.groupby(['timestamp','track_id']).filter(lambda x: len(x) >1)
    nps = (nps.groupby(['timestamp','track_id']).apply(lambda pts: (NearestPoint(pts.drop(['timestamp','track_id'],axis=1).to_numpy())))).reset_index(name = 'NN')
    
    tracks = trackstats.groupby(['timestamp','track_id'])

    nps_array = nps.NN[:]
    nn_distance = []
    for i in range(0,nps_array.size):
        nn_distance.append(nps_array[i][0][0][1])

    nn_df = pd.DataFrame(nn_distance)


    ax = sns.histplot(nn_df)

    ax.set(xlabel='Euclidian distances between points in labeled tracks [m]')

    plt.show()
    
    print('The mean distance of points in the tracks are', np.mean(nn_distance), 'and std of', np.std(nn_distance))


    #sns.scatterplot(data=groupedStats,x="range_sc", y = "rcs")




if __name__ == '__main__':
    main()
