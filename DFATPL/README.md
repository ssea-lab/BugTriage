# DFATPL--Dataset for "Empirical study on developer factors affecting tossing path length of bug reports"

## Reference

If you want to use the dataset or code, please cite [our paper](http://digital-library.theiet.org/content/journals/10.1049/iet-sen.2017.0159):

```
@inproceedings{title={Empirical study on developer factors affecting tossing path length of bug reports},
  authors={Hongrun Wu, Haiyang Liu, Yutao Ma},
  journal={IET Software},
  volume={12},
  number={3},
  pages={258–270},
  year={2018},
}
```

## Data description

DFATPL_Eclipse contains two files "history&tp" and "input_data", which are used in the experiments for the Eclipse project. Similarly, the file directory DFATPL_Mozilla contains the files used in the experiments for the Mozilla project.

```
1. history&tp -- the modified history and tossing path of bug reports.

(a) Bughistory_filter -- This file directory contains the modified history ("When," "Who," "What," "Added," and "Removed") of bug reports. It was obtained from "Bughistory.rar" by extracting the "What" field with "Assignee", "Status," and "Resolution."  

For example, the details of BugID=35 in the Eclipse project:
When                    Who             What    Added    Removed
2001-10-12 16:36:59 EDT	jean-michel_lemieux	Assignee	Michael_Valenta	Jean-Michel_Lemieux
2001-10-12 16:36:59 EDT	jean-michel_lemieux	Status	NEW	ASSIGNED
2001-10-18 16:21:36 EDT	Michael_Valenta	Status	ASSIGNED	NEW
2002-03-27 11:29:44 EST	Michael_Valenta	Assignee	Jean-Michel_Lemieux	Michael_Valenta
2002-03-27 11:29:44 EST	Michael_Valenta	Status	NEW	ASSIGNED
2002-03-27 11:29:44 EST	Michael_Valenta	Status	NEW	ASSIGNED
2002-04-03 14:44:18 EST	jean-michel_lemieux	Status	RESOLVED	NEW
2002-04-03 14:44:18 EST	jean-michel_lemieux	Resolution	FIXED	---
2002-04-09 09:16:09 EDT	jean-michel_lemieux	Status	VERIFIED	RESOLVED

(b) TossingPath.rar -- This file shows the tossing path of a bug report. Although all developers in a tossing path have participated in the fixing process of a bug report, the last developer in the path is the fixer who finally fixes the bug report. For example, in the file "8.csv" of the Eclipse project, the bug report ("BugID"=10407) is tossed out eight times until it is fixed.  

BugID	Dev1     Dev2  Dev3      Dev4          Dev5      Dev6        Dev7                Dev8
10407	kevin_haaland	mike_wilson	grant_gayed	nick_edgar	tod_creasey	kai-uwe_maetzel	platform-text-inbox	daniel_megert
13895	nick_edgar	randy_giffen	tod_creasey	chris_mclaren	csmclaren	debbie_wilson	platform-ui-inbox	bokowski
14856	eduardo_pereira	kevin_haaland	platform-ui-inbox	tod_creasey	mvm	michaelvanmeekeren	platform-swt-inbox	veronika_irvine
15370	dj_houghton	nick_edgar	eduardo_pereira	kai-uwe_maetzel	platform-text-inbox	eclipse	tom_eicher	daniel_megert
15655	erich_gamma	dirk_baeumer	daniel_megert	nick_edgar	randy_giffen	simon_arsenault	chris_mclaren	lynne_kues
16552	klicnik	dejan	nick_edgar	tod_creasey	susan_franklin	susan	platform-ui-triaged	remy.suen
17117	james_moody	nick_edgar	eduardo_pereira	kevin_haaland	platform-ui-inbox	tod_creasey	mvm	michaelvanmeekeren
17528	philippe_mulet	jerome_lanneluc	erich_gamma	adam_kiezun	akiezun	dj_houghton	platform-core-inbox	john_arthorne
18901	nick_edgar	randy_giffen	simon_arsenault	andrew_irvine	eduardo_pereira	airvine	platform-ui-inbox	tod_creasey
```

2. DFATPL_ProjectName_input: The input data for machine learning classfiers is included in this file directory. 

```
(a) feautres_input_TossingProb.csv -- The tossing probability between two developers are listed in this file.

(b) feautres_input.rar -- The feautres inputted to machine learning classfiers in our paper are included in this file.

(c) deldevs.csv -- This file includes developers who are on the very weak tossing ties.
```

## Additional details

The dataset used in this paper can be obtained by processing the [raw data](https://github.com/ssea-lab/BugTriage/tree/master/raw%20data).

## License
This dataset is licensed under the [BugTriage Project](https://github.com/ssea-lab/BugTriage/blob/master/LICENSE) license.
