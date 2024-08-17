import os
import numpy as np
import pandas as pd 
import seaborn as sns
from radar_scenes.sequence import Sequence
from radar_scenes.labels import Label, ClassificationLabel

from collections import Counter
from sklearn.cluster import dbscan
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

    
"""
def Frame_DBSCAN(X) :
    pts_array = np.transpose(np.array(X))
#    core_samples, labels = dbscan(pts_array, eps = 3, min_samples = 2)
    if (np.size(pts_array)>2) :
        hdb = HDBSCAN(min_cluster_size=2)
        hdb = hdb.fit(pts_array)
        labels = np.array(hdb.labels_)
    else :
        labels = np.array([-1])
    return pts_array, range(0,np.size(pts_array)), labels
"""

def Frame_DBSCAN(X) :
    pts_array = np.transpose(np.array(X))
    core_samples, labels = dbscan(pts_array, eps = 4, min_samples = 2)
    return pts_array, core_samples, labels

def NearestCenter(target) :
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(target)
    distances, indices = nbrs.kneighbors(target)
    return distances, indices


def Ronen_Scoring(tracks, clusters) :
    centers=[]
    alg_centers=[]
    centers_fit = set()

    trackCenters = np.transpose(np.array([tracks['x_cc']['mean'], tracks['y_cc']['mean']]))
    clusterCenters = np.transpose(np.array([clusters['x_cc']['mean'], clusters['y_cc']['mean']]))

    for real_center in trackCenters:
        for alg_center in clusterCenters:
            if (alg_center[0],alg_center[1]) in centers_fit:
                continue
            x = alg_center-real_center
            if(np.linalg.norm(x))<3:
              centers_fit.add((alg_center[0],alg_center[1]))
              break

    return centers_fit

def Scoring(tracks, clusters) :
    
#    tracksBB = np.empty(shape = (tracks.len,4))
#    tracksBB[:,0] =tracks['x_cc']['min']
#    [tracks['x_cc']['min'], tracks['y_cc']['min']], tracks['x_cc']['max'], tracks['y_cc']['max'])
#    clusterBB = [np.array([clusters['x_cc']['min'], clusters['y_cc']['min']]), np.array([clusters['x_cc']['max'], clusters['y_cc']['max']])]

    minBoxes = np.min(np.array((tracks['x_cc']['min'],clusters['x_cc']['min'])))

    for realBB in trackBB:
        for algBB in clusterBB:
            lx = np.max(realBB[0])
#            cumIoU = 
    return centers_fit



def main():
    # MODIFY THIS LINE AND INSERT PATH WHERE YOU STORED THE RADARSCENES DATASET
    path_to_dataset = "C:\\Users\\eric_\\Dropbox\\Schooling\\Eric\\Technion\\Project1\\RadarScenes\\data"

    if not os.path.exists(path_to_dataset):
        print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        return
    
    file_list = os.listdir(path_to_dataset)
    

    fullScoreList = []

    filenum = 0
    while filenum < len(file_list) :

        #filename = 'sequence' + str(filenum)
        # Define the *.json file from which data should be loaded
        seqpath = os.path.join(path_to_dataset, file_list[filenum])
        while (filenum < len(file_list)) and (not os.path.isdir(seqpath)) :
            filenum+=1
            seqpath = os.path.join(path_to_dataset, file_list[filenum])


        fullpath = os.path.join(seqpath, "scenes.json")

        if not os.path.exists(fullpath):
            print("Please modify this example so that it contains the correct path to the dataset on your machine.")
            return        

        # create sequence object from json file
        sequence = Sequence.from_json(fullpath)
        #radar_data = radar_data[radar_data["track_id"] != b'']

        trackstats = pd.DataFrame(sequence.radar_data)
    
        #filter out the static objects based on 3 m/s threshold
#        trackstats = trackstats.drop(trackstats[trackstats.vr_compensated < 3.].index)
#        trackstats = trackstats.drop(trackstats[trackstats.track_id == b''].index)

        #create a new cluster column with a default -1 value
        trackstats['Clusters'] = -1
        vr_weight = 1.69
        #create clusters with DBSCAN
        groupedClusters = trackstats[trackstats.vr_compensated > 3.].groupby(['timestamp']).apply(lambda pts: (Frame_DBSCAN([pts['x_cc'],pts['y_cc'],pts['vr_compensated']*vr_weight])),include_groups=False).reset_index(name = 'Clusters')
        #Add cluster column to trackstats by finding the cluster x,y points in the series and give a cluster number. 
        #The default cluster# is -1 and will remain if the point is static or doesnt belong to a cluster

        #assign a unique ID# in addition to the trackID string
        trackstats = trackstats.assign(track_num = trackstats['track_id'].astype('category').cat.codes)

        #Stats of the labeled points 
#        groupedStats = trackstats.groupby(['timestamp','track_id']).agg({'x_cc':['mean','min','max'],'y_cc':['mean','min','max']})
#        groupedStats['count'] = trackstats.groupby(['timestamp','track_id']).size()

        #    print(groupedClusters)


        scoreList = []
        for timeframe_idx, clustersInTimeframe in groupedClusters.iterrows() :
            #Set the cluster of each clustered point in the main database by the timestamp and x,y point
            for pt_idx in range(clustersInTimeframe['Clusters'][0].shape[0]) :
                trackstats.loc[(trackstats['timestamp'] == clustersInTimeframe['timestamp']) & 
                               (np.isclose(trackstats['x_cc'],clustersInTimeframe['Clusters'][0][pt_idx,0]) & 
                                np.isclose(trackstats['y_cc'],clustersInTimeframe['Clusters'][0][pt_idx,1])),'Clusters'] = clustersInTimeframe['Clusters'][2][pt_idx]
            
            rScore = adjusted_rand_score(trackstats[trackstats['timestamp'] == clustersInTimeframe['timestamp']]['Clusters'].to_numpy(), trackstats[trackstats['timestamp'] == clustersInTimeframe['timestamp']]['track_num'].to_numpy())

            scoreList.append([sequence.sequence_name, clustersInTimeframe['timestamp'], rScore])

#            print('Sequence:', sequence.sequence_name, ' Timestamp:', clustersInTimeframe['timestamp'], ' Rand Score: ', rScore)

            '''
            fig, ax = plt.subplots(1)
            ax.set_title(['Adjusted rand score: ', rScore])
            ax.scatter(trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['x_cc'],
                        trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['y_cc'], 
                        c=trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['Clusters'],
                        s = 40)
            
            ax.scatter(trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['x_cc'],
                        trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['y_cc'], 
                        c=trackstats[(trackstats['timestamp'] == clustersInTimeframe['timestamp'])]['label_id'],
                        marker = 'x')
                
            ax.grid(True)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
            '''

        dfScores = pd.DataFrame(scoreList, columns = ['Sequence Name', 'Timestamp', 'Rand Score'])
        print('Sequence processed:', sequence.sequence_name, ' Num of frames processed: ', len(dfScores.index), ' Avg Rand Score: ', dfScores['Rand Score'].mean())
        dfScores.to_csv(fullpath + sequence.sequence_name + 'scores_xy_vr.csv', index = False)
        fullScoreList.append([sequence.sequence_name,len(dfScores.index),dfScores['Rand Score'].mean()])  #put the sequence name, num frames processed and mean rand score in the full score list
        filenum+=1


    dfScores = pd.DataFrame(fullScoreList, columns = ['Sequence Name', '# Frames processed', 'Avg Rand Score'])
    print('Num of sequences processed: ', len(dfScores.index), ' Avg Rand Score: ',dfScores['Rand Score'].mean())
    dfScores.to_csv(path_to_dataset + 'total_scores_xy_vr.csv', index = False)
    
    """
    cluster_array = groupedStats.Clusters[0][0]
    labels = groupedStats.Clusters[0][2]
    for scene_idx in range(1,groupedStats.Clusters.size) :
        //aggregate the points
        cluster_array = np.append(cluster_array, groupedStats.Clusters[scene_idx][0], axis =0)
        //aggregate the labels
        labels = np.append(labels,groupedStats.Clusters[scene_idx][2],axis = 0)
    labels = labels[:,np.newaxis]
    cluster_array = np.append(cluster_array,labels,axis = 1)
    # remove the noisy points
    """
    



    """
    fig, ax = plt.subplots(2,2)

    scene_idx = 4

    cluster_array = groupedStats.Clusters[scene_idx][0]
    labels = groupedStats.Clusters[scene_idx][2]
    labels = labels[:,np.newaxis]
    cluster_array = np.append(cluster_array,labels,axis = 1)
    cluster_array = cluster_array[cluster_array[:,-1] >= 0]
    ax[0,0].scatter(cluster_array[:,0],cluster_array[:,1], c=cluster_array[:,2],label=cluster_array[:,2])
    ax[0,0].set_xlim(-100,100)
    ax[0,0].set_ylim(-100,100)
    ax[0,0].grid(True)

    cluster_array = groupedStats.Clusters[scene_idx+1][0]
    labels = groupedStats.Clusters[scene_idx+1][2]
    labels = labels[:,np.newaxis]
    cluster_array = np.append(cluster_array,labels,axis = 1)
    cluster_array = cluster_array[cluster_array[:,-1] >= 0]
    ax[0,1].scatter(cluster_array[:,0],cluster_array[:,1], c=cluster_array[:,2],label=cluster_array[:,2])
    ax[0,1].set_xlim(-100,100)
    ax[0,1].set_ylim(-100,100)   
    ax[0,1].grid(True)

    cluster_array = groupedStats.Clusters[scene_idx+2][0]
    labels = groupedStats.Clusters[scene_idx+2][2]
    labels = labels[:,np.newaxis]
    cluster_array = np.append(cluster_array,labels,axis = 1)
    cluster_array = cluster_array[cluster_array[:,-1] >= 0]
    ax[1,0].scatter(cluster_array[:,0],cluster_array[:,1], c=cluster_array[:,2],label=cluster_array[:,2])
    ax[1,0].set_xlim(-100,100)
    ax[1,0].set_ylim(-100,100)
    ax[1,0].grid(True)

    cluster_array = groupedStats.Clusters[scene_idx+3][0]
    labels = groupedStats.Clusters[scene_idx+3][2]
    labels = labels[:,np.newaxis]
    cluster_array = np.append(cluster_array,labels,axis = 1)
    cluster_array = cluster_array[cluster_array[:,-1] >= 0]
    ax[1,1].scatter(cluster_array[:,0],cluster_array[:,1], c=cluster_array[:,2],label=cluster_array[:,2])
    ax[1,1].grid(True)
    ax[1,1].set_xlim(-100,100)
    ax[1,1].set_ylim(-100,100)   
    plt.show()
    """




if __name__ == '__main__':
    main()
