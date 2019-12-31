# BugTriage

This work was supported by the National Natural Science Foundation of China under Grant No. 61272111. To carry out the study of automated bug triage, we retrieved and collected 200,000 and 220,000 fixed bug reports of the Eclipse project and the Mozilla project, respectively. For researchers who are interested in this dataset, you can feel free to download and use it. Also, if you think that it is useful for your work, please help cite the following papers. 

- 刘海洋, 马于涛. [一种针对软件缺陷自动分派的开发者推荐方法](http://xwxt.sict.ac.cn/CN/Y2017/V38/I12/2747). 小型微型计算机系统, 38(12): 2747-2753, 2017. (For papers written in Chinese)
- Hongrun Wu, Haiyang Liu, Yutao Ma. [Empirical study on developer factors affecting tossing path length of bug reports](https://ieeexplore.ieee.org/document/8371790). IET Software, 12(3): 258-270, 2018. (For papers written in English)

In addition to traditional machine learning algorithms (such as SVM), we implemented deep learning algorithms based on convolutional neural networks (CNNs). If you want to compare your own approach with these CNN-based algorithms (as baseline approaches), please refer to [the code](https://github.com/huazhisong/graduate_text) written by [Huazhi Song](https://github.com/huazhisong) and cite the following paper.

- 宋化志, 马于涛. [DeepTriage：一种基于深度学习的软件缺陷自动分配方法](http://xwxt.sict.ac.cn/CN/Y2019/V40/I1/126). 小型微型计算机系统, 40(1):126-132, 2018. --Huazhi Song, Yutao Ma. DeepTriage: An Automatic Triage Method for Software Bugs Using Deep Learning. Journal of Chinese Computer Systems, 40(1):126-132, 2018. (in Chinese with English Abstract)

The study mentioned above focuses on predicting the final fixer for a given bug report. Another view holds that any developer on the tossing path of a bug report contributes to the resolution of the bug. Recently, a few researchers considered bug triage as a multi-label classification problem. To this end, we also provide a dataset called [MLBT](https://github.com/ssea-lab/BugTriage/tree/master/MLBT) for researchers who are working on this problem. If you want to use this dataset and the benchmark result of our approach, please cite the following thesis.

- 史小婉. [一种基于文本分类和评分机制的软件缺陷自动分派方法研究](https://github.com/ssea-lab/BugTriage/blob/master/MLBT/一种基于文本分类和评分机制的软件缺陷自动分派方法研究.pdf). 武汉: 武汉大学, 2018. (Xiaowan Shi. A Software Bug Triaging Method Based on Text Classification and Developer Rating. Wuhan: Wuhan University, 2018.)
