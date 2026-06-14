文件使用说明
mp3d_view.py：用于查看环境，放置物品，其中放置物品的配置文件时object_config.yaml
    每次移动图片都会保存到observations文件夹下面
seg.py:用于计算每个像素点和输入文本的相似度，会将结果输出到score_data文件夹下面的scores_matrix.csv
        运行时需要调用observations文件夹下面的图片，目前只能调用一张图片
loss.py:对目标进行优化的方法函数