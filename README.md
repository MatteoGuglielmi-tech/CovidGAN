<!--toc:start-->

- [:rotating_light: Problem statement :rotating_light:](#rotatinglight-problem-statement-rotatinglight)
- [:interrobang: Why? :interrobang:](#interrobang-why-interrobang)
  - [Figure 1](#figure-1)
- [:floppy_disk: Data :floppy_disk:](#floppydisk-data-floppydisk)
- [:running: Execution :running:](#running-execution-running)
  - [:heavy_plus_sign: Dependencies :heavy_plus_sign:](#heavyplussign-dependencies-heavyplussign)
  - [:rocket: Run the program :rocket:](#rocket-run-the-program-rocket)
  - [:chart_with_downwards_trend: Visualize losses from different runs :chart_with_downwards_trend:](#chartwithdownwardstrend-visualize-losses-from-different-runs-chartwithdownwardstrend)
- [:bar_chart: Results :bar_chart:](#barchart-results-barchart)
  - [Generator progression for score 0:](#generator-progression-for-score-0)
  - [Generator progression for score 1:](#generator-progression-for-score-1)
  - [Generator progression for score 2:](#generator-progression-for-score-2)
  - [Generator progression for score 3:](#generator-progression-for-score-3)
- [:wrench: Tools and PL used in this project :wrench:](#wrench-tools-and-pl-used-in-this-project-wrench)
<!--toc:end-->

## :medical_symbol: Synthetic ultrasound images generator :medical_symbol:

#### :rotating_light: Problem statement :rotating_light:

This project consists in devising a method that allows the **generation of synthetic medical images**.

#### :interrobang: Why? :interrobang:

The underlying idea was moved by the following aspects:

- very often, the dataset at disposal is, by nature, strongly unbalanced (Fig. 1) and, perhaps, generating synthetic images may allow **achieving a more balanced dataset**;
- **synthetic images** might be a key in emergency scenarios as they may help in **speeding up the development of automatic methods** for diagnostic and containment purposes;
- this generative approach can be extended to a wide variety of cases, without being specifically limited to a single application.

###### Figure 1

<div id="dist" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/score_distribution.png" /> 
</div>

#### :floppy_disk: Data :floppy_disk:

This method has been **developed and tested on a COVID-19 dataset** of positive patients provided by the [San Matteo hospital](http://www.sanmatteo.org/site/home.html).

#### :running: Execution :running:

##### :heavy_plus_sign: Dependencies :heavy_plus_sign:

Firstly, to be able to run the source code, it is necessary to install all the required dependencies and creating the required folders. This can be achieved by running the following command being inside the _GANs/_ folder:

> `chmod u+x ./src/set-up/setup.sh & ./src/set-up/setup.sh`

Side note: either conda or pip needs to be installed.

##### :rocket: Run the program :rocket:

To get a full list of the available parameters, run the following command:

> `python main.py -h`

to trigger the help interface.

To run the program, simply call the python interpreter on `main.py` with the necessary parameters.

##### :chart_with_downwards_trend: Visualize losses from different runs :chart_with_downwards_trend:

Losses of different runs are saved as a sequence of scalars in the `all_runs` folder. To graphically visualize their trend run the following command:

> `tensorboard --logdir=src/all_runs`

#### :bar_chart: Results :bar_chart:

(the rendering may take a while...)

###### Generator progression for score 0:

<div id="score0" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-0.gif" /> 
</div>

###### Generator progression for score 1:

<div id="score1" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-1.gif" /> 
</div>

###### Generator progression for score 2:

<div id="score2" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-2.gif" /> 
</div>

###### Generator progression for score 3:

<div id="score3" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-3.gif" /> 
</div>

#### :wrench: Tools and PL used in this project :wrench:

<a href="https://anaconda.com/" target="_blank">
    <img align="left" src="https://github.com/devicons/devicon/blob/master/icons/anaconda/anaconda-original-wordmark.svg" alt="anaconda" height="40px"/> 
</a>
<a href="https://www.python.org/" target="_blank"> 
    <img align="left" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="python" height="40px"/>
</a> 
<a href="https://pytorch.org/" target="_blank">
    <img align="left" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="pytorch" height="40px"/>
</a>
<a href="http://www.gnu.org/software/bash/" target="_blank">
    <img align="left" src="https://github.com/devicons/devicon/blob/master/icons/bash/bash-original.svg" alt="bash" height="40px"/>
</a>
