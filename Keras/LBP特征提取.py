radius = 1;
n_point = radius * 8;

def texture_detect():
    train_hist = np.zeros( (200,256) );
    test_hist = np.zeros( (160,256) );
    for i in np.arange(200):
        #使用LBP方法提取图像的纹理特征.
        lbp=skft.local_binary_pattern(train_data[i],n_point,radius,'default');
        #统计图像的直方图
        max_bins = int(lbp.max() + 1);
        #hist size:256
        train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));

    for i in np.arange(160):
        lbp = skft.local_binary_pattern(test_data[i],n_point,radius,'default');
        #统计图像的直方图
        max_bins = int(lbp.max() + 1);
        #hist size:256
        test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));


    return train_hist,test_hist;