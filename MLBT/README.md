# MLBT--Dataset for "一种基于文本分类和评分机制的软件缺陷分配方法"

## Reference

If you want to use this dataset, please cite the following master thesis:

```
@thesis{
  title={一种基于文本分类和评分机制的软件缺陷分配方法}/{A Software Bug Triaging Method Based on Text Classification and Developer Rating},
  author={史小婉}/{Xiaowan Shi},
  school={武汉大学}/{Wuhan University},
  year={2018},
}
```

## Data description

MLBT contains two file directories "eclipse" and "mozilla", which are two datasets for the experiments on the Eclipse project and Mozilla project, respectively. 

```
1. features_component_product.rar  -- it records the extracted features (i.e., component and product) of each bug report.

2. Buginfo -- this file directory contains the infomation (i.e., summary, description, and comments) and fixer of each bug report.

3. multi-labels.rar -- this file contains all the participants on the tossing path of each fixed bug report. (Note that any developer on the tossing path of a bug report contributes to the resolution of the bug.)
```

## Additional details

The dataset used in this thesis can be obtained from the [raw data](https://github.com/ssea-lab/BugTriage/tree/master/raw%20data).

## License

This dataset is licensed under the [BugTriage Project](https://github.com/ssea-lab/BugTriage/blob/master/LICENSE) license.

