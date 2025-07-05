# Divisible Configuration Performance Learning
>Machine/deep learning models have been widely adopted for predicting the configuration performance of software systems.
However, a crucial yet unaddressed challenge is how to cater for the **sparsity** inherited from the configuration landscape:
the influence of configuration options (features) and the distribution of data samples are highly sparse.
>
> In this paper, we propose a model-agnostic and sparsity-robust framework for predicting configuration performance, dubbed **DaL**, 
based on the new paradigm of dividable learning that builds a model via "divide-and-learn". To handle sample sparsity, 
the samples from the configuration landscape are divided into **distant divisions**, for each of which we build a sparse local model, 
e.g., **regularized Hierarchical Interaction Neural Network**, to deal with the feature sparsity. A newly given configuration would then be assigned 
to the right model of division for the final prediction. Further, *DaL* adaptively determines the **optimal number of divisions** 
required for a system and sample size without any extra training or profiling. 
>
> Experiment results from 12 real-world systems and five sets of training data reveal that, compared with the state-of-the-art approaches, *DaL* performs no worse than 
the best counterpart on 44 out of 60 cases (within which 31 cases are significantly better) with up to *1.61×* 
improvement on accuracy; requires fewer samples to reach the same/better accuracy; and producing acceptable training 
overhead. In particular, the mechanism that adapted the parameter *d* can reach the optimal value for 76.43% of the 
individual runs. The result also 
confirms that the paradigm of dividable learning is more suitable than other similar paradigms such as ensemble learning 
for predicting configuration performance. Practically, *DaL* considerably improves different global models when using 
them as the underlying local models, which further strengthens its flexibility.
> To promote open science, all the data, code, and supplementary materials of this work can be accessed at our repository: https://github.com/ideas-labo/DaL-ext.
 
This repository contains the **key codes**, **full data used**, and **raw experiment results** for the paper.

# Documents
- **data**:
performance datasets of 12 subject systems as specified in the paper.

- **results**:
contains the raw experiment results for all the research questions.

- **supplementary_materials**：
contains all the supplementary files for the paper, such as the full list of systematic literature review studies that are related to sparsity, as mentioned in the paper.

- **utils**
contains utility functions to build DNN models.

- **DaL_main.py**: 
the *main program* for using DaL, which automatically reads data from csv files, trains and evaluates, and save the results.

- **requirements.txt**:
the necessary packages required to run the program.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python and Tensorflow:
the codes have been tested with **Python 3.6 - 3.9** and **Tensorflow 2.12 - 2.15**, other versions might cause errors.

3. Install all missing packages according to **requirements.txt** and runtime messages.


# Run *DaL*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python DaL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *DaL_main.py* file on the IDE, and simply click 'Run'.


# Demo Experiment
The main program *DaL_main.py* defaultly runs a demo experiment that evaluates *DaL* with 5 sample sizes of *Lrzip*, 
each repeated 30 times, without hyperparameter tuning (to save demonstration time).

A **successful run** would produce similar messages as below: 

    ---Lrzip, Run 1, Size 1, Training size: 127, Testing size: 5057---
    > Adapting d...
        >> Selected depth: 1
        >> Adapting time cost (minutes): 0.0005771319071451823
    > Dividing...
        >> Number of divisions: 2
        >> Dividing time cost (minutes): 0.00040945212046305336
    > Training RF classifier...
        >> Classifying testing samples
    > Training Local models...
        >> Tuning division 1... (23 samples)
        >> Tuning division 2... (104 samples)
        >> Learning division 1... (23 samples)
        >> Learning division 2... (104 samples)
    > Testing...
        >> DaL MRE: 23.37
        >> Training time (min): 0.08145502408345541
        >> Predicting time (min) : 0.24003959496816002
    Total time (min) : 0.3219040711720785

The results will be saved in a file in the same directory with a name in the format *'DaL_System_d_N-Train_N-Test_Seed.csv'*, for example, *'DaL_Lrzip_d1_127-5057_2.csv'*.

# Change Experiment Settings
To run more complicated experiments, alter the codes following the instructions below and comments in *DaL_main.py*.

#### To switch between subject system(s)
    Modify line 21 following the comments in DaL_main.py.

    E.g., to run DaL with Apache and BDBC, set line 21 to 'selected_datasets = [0, 1]'.


#### To evaluate using different sample size(s)
    For example, to run S_1 and S_2, set line 22 to 'selected_sizes = [0, 1]'.


#### To save the experiment results
    Set 'save_results = True' at line 23.
    
    
#### To tune the hyperparameters (takes longer time)
    Set line 24 with 'test_mode = False'.


#### To change the number of experiments 
    Change 'start_run' and 'end_run' at line 27 and 26, where N_experiment is end_run-start_run. 


#### To fix/adapt the depth *d*
    To fix the depth to 1, modify line 31 to 'depth_selection_mode = ['fixed-1']'
    
    To adapt the depth using averaging hypervolume as proposed in the paper, modify line 31 to 'depth_selection_mode = ['AvgHV']'




# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *DaL* in the paper. 

- [HINNPerf](https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh)

    A novel deep learning performance model that utilizes hierarchical interaction neural network to achieve better prediction accuracy.

- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.

- [DECART](https://github.com/jmguo/DECART)

    CART with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature selection.

- [Perf-AL](https://github.com/GANPerf/GANPerf)

    Novel GAN-based performance model with a generator to predict performance and a discriminator to distinguish the actual and predicted labels.
    

To compare *DaL* with other SOTA models, please refer to their original pages (you might have to modify or reproduce their codes to ensure the compared models share the same set of training and testing samples).
