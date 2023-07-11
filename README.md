<!--toc:start-->

- [:rotating_light: Problem statement :rotating_light:](#rotatinglight-problem-statement-rotatinglight)
- [:interrobang: Why? :interrobang:](#interrobang-why-interrobang)
- [:floppy_disk: Data :floppy_disk:](#floppydisk-data-floppydisk)
- [:running: Execution :running:](#running-execution-running)
  - [:heavy_plus_sign: Dependencies :heavy_plus_sign:](#heavyplussign-dependencies-heavyplussign)
  - [:rocket: Run the program :rocket:](#rocket-run-the-program-rocket)
  - [:chart_with_downwards_trend: Visualize loss trend across different runs :chart_with_downwards_trend:](#chartwithdownwardstrend-visualize-loss-trend-across-different-runs-chartwithdownwardstrend)
- [:bar_chart: Results :bar_chart:](#barchart-results-barchart)
- [:wrench: Tools and PL used in this project :wrench:](#wrench-tools-and-pl-used-in-this-project-wrench)
<!--toc:end-->

## :medical_symbol: Synthetic ultrasound images generator :medical_symbol:

#### :rotating_light: Problem statement :rotating_light:

This project is about devising a method that allows to **generate synthetic medical images**.

#### :interrobang: Why? :interrobang:

- in certain cases, the dataset at disposal is, by nature, strongly unbalanced. Perhaps, generating synthetic images allows **to achieve an homogeneously distributed dataset**;
- **synthetic images** might be a key in emergency scenarios. Indeed, it may help in **speeding up the development of automatic methods** for diagnostic and containment purposes.
- this generative approach can be extended to a variety of cases, without being specifically tied to a single application.

#### :floppy_disk: Data :floppy_disk:

This method has been **developed and tested upon COVID-19 positive patients** provided by the [San Matteo hospital](http://www.sanmatteo.org/site/home.html).

#### :running: Execution :running:

##### :heavy_plus_sign: Dependencies :heavy_plus_sign:

To be able to run the source code it is first necessary to install the require dependencies. In particular, this operation can be performed either via pip or anaconda. To do so, run one of the following commands:  
`pip install -r dependencies.yml`  
or  
`conda env create -f dependencies.yml`

##### :rocket: Run the program :rocket:

To get a full list of custom parameters run the following command:  
`python main.py -h`  
to trigger the help interface.

To run the program, simply call the python interpreter on `main.py` with the necessary parameters and everything will start running.

##### :chart_with_downwards_trend: Visualize loss trend across different runs :chart_with_downwards_trend:

Losses across different runs are saved as a sequence of scalars in the `runs` folder. To graphically visualize their trend run the following command:  
`tensorboard --logdir=runs`

#### :bar_chart: Results :bar_chart:

<div id="gifs" align="center">
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-0.gif" width="500" height="500" /> 
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-1.gif" width="500" height="500" /> 
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-2.gif" width="500" height="500" /> 
   <img src="https://github.com/MatteoGuglielmi-tech/CovidGAN/blob/main/Images/generator_progress_score-3.gif" width="500" height="500" /> 
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
