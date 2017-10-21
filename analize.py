import numpy as np
import time
from config import  conf
import os
import cv2

ANGLE = 0
CLASS = 1
OCTAVE = 2
PTX = 3
PTY = 4
RESPONSE = 5
SIZE = 6

timestamp = time.strftime(conf.time_format_day, time.gmtime())

def get_stats_for_day():
    statistics = {}
    for data_file in os.listdir(conf.data_folder):
        data_stamp = data_file[:-4]
        filename = conf.data_folder + data_file
        image = cv2.imread(conf.image_folder + data_stamp + '.jpg')
        if data_file.startswith(timestamp):
            data = np.load(filename)
            stats = {}
            #stats
            stats['mean_size'] = np.mean(data[:,SIZE])
            stats['max_size'] = np.max(data[:,SIZE])
            stats['min_size'] = np.min(data[:,SIZE])
            stats['area'] = np.sum(data[:,SIZE])
            stats['area_percentage'] = stats['area']/image.size * 100
            stats['n_blobs'] = float(len(data[:,SIZE]))
            statistics[data_stamp] = stats
    return statistics


def format_stats(stats):
    report = ""
    keys = [stat for stat in stats]
    keys.sort()
    for stat in keys:
        report += "{}\n{}\n{}\n".format(("-"*len(stat)), stat, ("-"*len(stat)))
        for feature in stats[stat]:
            report+= "{}:{:.2}\n".format(feature, stats[stat][feature])
    return report

stats = get_stats_for_day()
report = format_stats(stats)
f = open(conf.reports_folder + timestamp + '.txt', 'w')
f.write(report.rstrip())
f.close()
