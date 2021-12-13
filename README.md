# ids-ex1 - Final Review Branch

This is the Final Review branch for the first assignment of IDS WS2021/2022. We have modified the conda environment to install:

 * [pandarallel](https://github.com/nalepae/pandarallel) to utilitize multicore architechtures when using pandas apply
 * [category_encoders](https://contrib.scikit-learn.org/category_encoders/) as easy-to-use library to encode categorial features


To use the environment 

```
conda env create --name ids-1 --file=env-IDS2021_G31.yaml
conda activate ids-1
``` 

Maintainers: 
 * **Lorraine Saju** @lorraine.saju
 * **Nahn Le** @nhantle
 * **Giang Nguyen** @gvnguyen


## About the code

Since our group consist of 3 people, the code base will sometimes differ in style! We hope this is not to hard to read. Also we will provide necessary explainations as comments or in special markdown blocks. Simple task on the other hand are not explained as code alone should be sufficient.

## Project structure

```
├── assignment_part1.ipynb
├── data
├── dataset.csv
├── env-IDS2021_G31.yaml
├── flights_classifying.csv
├── models
├── README.md
└── visuals                 

3 directories, 6 files

```

Explanation
```
├── python jupyter notebook
├── directory containing all intermediate datasets
├── originial data set .csv
├── conda environment .yaml
├── original flights classifying .csv
├── directory containing all intermediate / classifications models
├── README.md
└── directory containing all visuals

```

