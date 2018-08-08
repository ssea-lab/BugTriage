# MLC--Dataset for "一种基于文本分类和评分机制的软件缺陷分配方法"

## References

If you use our dataset or code, please cite [our paper](https://github.com/ssea-lab/BugTriage/tree/master/raw%20data/eclipse):

```
@thesis{
  title={一种基于文本分类和评分机制的软件缺陷分配方法},
  author={史小婉},
  school={武汉大学},
  year={2018},
}
```

## Data description

MLC contains two files "eclipse" and "mozilla", which are the datasets for experiments of Eclipse project and Mozilla project in the paper, respectively. 

```
1. com-pro-id.csv -- it shows the extracted component and product feature of each bug report.
2. Buginfo -- this file directory contains the infomation (summary, description and comments) and the fixer of bug reports
3. multilabels.rar -- this dataset contains the multi-labels (i.e., all the participants) of each fixed bug report.
```

## Additional details

The datasets in this paper can be obtained from the raw data in https://github.com/ssea-lab/BugTriage/tree/master/raw%20data/eclipse.

## License

This dataset is licensed under the [BugTriage Project](https://github.com/ssea-lab/BugTriage/blob/master/LICENSE) license.

