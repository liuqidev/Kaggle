import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import getopt
import pickle
import time


def basic_details(df):
    print('Row:{0} Column:{1}'.format(df.shape[0], df.shape[1]))
    d = pd.DataFrame()
    d['Number of unique'] = df.nunique()
    d['Number of missing'] = df.isnull().sum()
    d['Datatype'] = df.dtypes

    return d


def load_review(lo, hi):
    lo = int(lo)

    review={}
    # 取记录中的第lo, hi)半开半闭区间行
    N=100 # 文件总共行数,是个固定值
    path= "../Datasets/yelp-dataset/"
    # print(os.listdir(path))
    if hi=='' or hi==None:
        review['data'] = pd.read_csv(path + "yelp_review.csv", header=0, skiprows=range(0, lo), nrows=50)
        hi=review['data'].shape[0]
    else:
        hi = int(hi)
        review['data'] = pd.read_csv(path+"yelp_review.csv", header=0, skiprows=range(0, lo), nrows=hi-lo)
    review['left']=lo
    review['right']=hi
    return review


def plot_stars(review):
    plt.figure(figsize=(12,8))
    review_stars=review['data']['stars']
    ax = sns.countplot(review_stars)
    plt.ylabel("Count",fontsize=16)
    plt.xlabel("Stars",fontsize=16)
    plt.title('Distribution of stars', fontsize=16)
    plt.savefig('distibution_of_stars_{}_to_{}.png'.format(review['left'], review['right']))
    plt.show()


# 使用命令行操作，便于将脚本进行并行以及集群操作
def main(argv):
    left=''
    right=''
    output=''
    # python review.py -l 0 -r 10 -o output.png
    try:
        opts, args = getopt.getopt(argv, "hl:r:o:", ["outfile"])
    except getopt.GetoptError:
        print("Error: review.py -l <left> -r <right> -o <outputfilename>")
        print("       or review.py -left <left> --right <right> -ouput <outputfilename>")
        sys.exit(2)

    for opt, arg in opts:
        if opt=="-h":
            # help
            print("Error: review.py -l <left> -r <right> -o <outputfilename>")
            print("   or: review.py -left <left> --right <right> -ouput <outputfilename>")
            sys.exit()

        elif opt in ("-l", "--left"):
            left=arg
        elif opt in ("-r", "--right"):
            right=arg
        elif opt in ("-o", "--output"):
            output=arg


    review=load_review(lo=left, hi=right)
    review_df =review['data']
    left=review['left']
    right=review['right']
    print("Left:", left)
    print("Right:", right)
    print("Output:", output)
    details_df=basic_details(review_df)
    print(details_df)
    with open("details_df_{}_to_{}.pkl".format(left, right), "wb") as savefile:
        pickle.dump(details_df, savefile)

    # plot_stars(review)

    # 把频率统计结果存入文件中
    review_stars = review['data']['stars']
    count_star={}
    # star_list=review_stars.unique()
    count = {}
    for i in review_stars:
        if i not in count.keys():
            count[i] = 1
        else:
            count[i] += 1

    with open("stars_count_{}_to_{}.txt".format(left, right), "w") as f:
        total=0
        for k in sorted(count.keys()):
            total+=count[k]
            print("Stars {}: {}".format(k, count[k]))
            f.write("Stars {}: {}\n".format(k, count[k]))
        print("Total: {}".format(total))
        f.write("Total: {}\n".format(total))

    end_t = time.time()
    timecost="Running time: {} s".format(end_t - star_t)
    print(timecost)
    with open("timecost_{}_to_{}.txt".format(left, right), "w") as timecostfile:
        timecostfile.write(timecost)


# 所有参数范围都是半开半闭区间
if __name__=="__main__":
    # print(sys.argv)
    star_t = time.time()
    main(sys.argv[1:])


