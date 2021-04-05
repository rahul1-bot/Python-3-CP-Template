#! /home/rahul/Desktop/_Python/PyTorch_Projects/torch_4.py
from __future__ import annotations
__name__: typing.__name__ = '__main_module__'

# --> Module: __main_module__ :: Modularity Code 
# DataSet Class 
# Data Analysis Class
# Data Preprocess Class 
# HardWare Config Class -> Google's TPU | GPU
# HyperParameter Class
# DL Model Class Training/test Loop 
# Evaluation Class
# ..
# Jypyter NoteBook: 
# --> from __main_module__ import * 

# ------- Py Imports
import os, collections, warnings, numba, re
from collections import defaultdict, deque
import typing
from typing import Container, NewType, Any
from numpy.core import numeric
warnings.filterwarnings(action= "ignore")

# ------ Data Analysis
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import iterable
import pandas as pd
pd.set_option("display.max_columns", 100)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats

# ----- Scripting
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectPercentile

# ---- PyTorch  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# --- Torch TypeScripts
Figure = NewType('Figure', Any)
Path = NewType('Path', Any)
Dir = NewType('Dir', Any)
Loader = NewType('Loader', Any)
File = NewType('File', Any)
Text = NewType('Text', Any)
Image = NewType('Image', Any)
Model = NewType('Model', Any)
Data = NewType('Data', Any)
Loss = NewType('Loss', Any)
Optimizor =  NewType('Optimizor', Any) 
Predictor =  NewType('Predictor', Any)

# --- DATASET ---------  
class Data(Dataset):
    def __init__(self, csv_file: File) -> None:
        self.dataset = pd.read_csv(csv_file, delimiter=",")
        self.x = self.dataset.iloc[::-1]
        self.y = self.dataset.iloc[:-1]


    def __repr__(self) -> tuple[str|Any, ...]:
        return self.__module__, type(self).__name__, hex(id(self))


    def __str__(self) -> dict[str, str]:
        info: list[str] = ["module", "name", "ObjectID"]
        return {item: value for item, value in zip(info, self.__repr__())} 


    def __len__(self) -> int:
        return self.dataset.shape[0]

    
    def __getitem__(self, index: int) -> tuple[pd.DataFrame|Any, ...]:
        return self.x[index], self.y[index]



# --- Data Analysis ----- 
class DataAnalysis:
    def __init__(self, dataset: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.dataset = dataset
        self.x = x
        self.y = y

    
    def __repr__(self) -> tuple[str|Any, ...]:
        return self.__module__, type(self).__name__, hex(id(self))


    def __str__(self) -> dict[str, str]:
        info: list[str] = ["module", "name", "ObjectID"]
        return {item: value for item, value in zip(info, self.__repr__())} 


    def dataCharacteristics(self, info: bool = False, summary: bool = False) -> Text:
        print(f"Shape of Dataset : {self.dataset.shape}")
        print(f"No. of Columns in DataSet : {self.dataset.shape[1]}")
        print(f"No. of Rows in DataSet : {self.dataset.shape[0]}")
        numericFeatures: Data = self.dataset.select_dtypes(include = [np.number])
        categoricalFeatures: Data = self.dataset.select_dtypes(exclude = [np.number])
        print(f"Number of Numeric Features : {numericFeatures.shape[1]}")
        print(f"Number of Categorical Features : {categoricalFeatures.shape[1]}")
        if info:
            print(self.dataset.info(verbose = False, memory_usage = "deep"))
        if summary:
            print(self.dataset.describe(include = "all", percentiles = [.15, .25, .50, .75, .85]).transpose())
        


    def nullPlot(self) -> Figure|Text:
        nullPercentage: Data = (self.x.isnull().sum() / len(self.x)) * 100
        try:
            nullPercentage = round(nullPercentage.drop(nullPercentage[nullPercentage == 0].index)).sort_values(ascending = False)
            plt.figure(figsize= (14, 10))
            nullplot: Figure = sns.barplot(x= nullPercentage.index, y= nullPercentage)
            plt.xticks(rotation= "90")
            plt.title("Percentage if Null Values in Training Set")
            plt.show()
            return nullplot
        except:
            return "No Null Values in the Training Set"



    def Anova(self, target: str) -> pd.DataFrame:
        categoricalFeatures: Data = self.x.select_dtypes(exclude= [np.number]).columns
        x_Copy: Data = self.x.copy()
        x_Copy[categoricalFeatures] = x_Copy[categoricalFeatures].fillna("missing")
        anova: dict[str, list[Data|Any]] = {"feature": [], "f": [], "p": []}
        for cat in x_Copy[categoricalFeatures]:
            groupPrices: list[Data] = []
            for group in x_Copy[cat].unique():
                groupPrices.append(x_Copy[x_Copy[cat] == group][target].values)

            f, p = stats.f_oneway(*groupPrices)
            anova["feature"].append(cat)
            anova["f"].append(f)
            anova["p"].append(p)
        anova: pd.DataFrame = pd.DataFrame(anova)
        anova = anova[["feature", "f", "p"]]
        anova.sort_values("p", inplace= True)
        return anova 
    

    def MutualInformation(self) -> Figure:
        numerics: list[str] = ["int16", "int32", "int64", "float16", "float32", "float64"]
        numeric_vars: list[Any] = list(self.x.select_dtypes(include= numerics).columns)
        x: pd.DataFrame = self.x[numeric_vars]
        y: pd.DataFrame = self.y.copy()
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 0)
        mi: pd.DataFrame = mutual_info_regression(x_train.fillna(0), y_train)
        mi: pd.Series = pd.Series(mi)
        mi.index = x_train.columns
        mi.sort_values(ascending= False).plot.bar(figsize= (18, 5))
        features: Any = SelectPercentile(mutual_info_regression, percentile= 10).fit(x_train.fillna(0), y_train)
        return x_train.columns[features.get_support()]


    
    def target_distribution(self) -> Figure:
        plt.figure(figsize = (12,8))
        plot1: Figure = sns.distplot(self.y , fit = stats.norm)
        plt.title("Target Distribution")
    
        mu, sigma = stats.norm.fit(self.y)
        plt.legend(["Normal dist. ($/mu=$ {:.2f} and $/sigma=$ {:.2f})".format(mu, sigma)], loc="best")
  
        fig: Figure = plt.figure()
        plot2 = stats.probplot(self.y, plot = plt)
        plt.show()
        print(plot1)
        print(plot2)
    


    def log_distribution(self) -> Figure:
        y: np.array[int] = np.log(self.y)
        plt.figure(figsize = (12,8))
        plot1 = sns.distplot(y , fit = stats.norm)
        plt.title("Target Distribution")

        mu, sigma = stats.norm.fit(y)
        plt.legend(["Normal dist. ($/mu=$ {:.2f} and $/sigma=$ {:.2f})".format(mu, sigma)], loc="best")
 
        fig: Figure = plt.figure()
        plot2 = stats.probplot(y, plot = plt)
        plt.show()
        print(plot1)
        print(plot2)



    def correration_coefficient(self) -> Figure:
        numeric_features = self.dataset.select_dtypes(include = [np.number])
        corr_matrix: Figure = numeric_features.corr()
        sns.set(style = "white")
        mask: np.array[int] = np.triu(np.ones_like(corr_matrix, dtype = np.bool))
        f, ax = plt.subplots(figsize = (20, 10))
        cmap: Figure = sns.diverging_palette(220, 1, as_cmap=True)
            
        sns.heatmap(data = corr_matrix, 
                    mask=mask, 
                    cmap=cmap, 
                    vmax=.3, 
                    center=0,
                    square=True, 
                    linewidths=.5, 
                    cbar_kws={"shrink": .5})
            
        plt.title("Mutual-Correration Plot")
        plt.show()


    
    def select_correration(self) -> Text:
        corr_set: set[Any] = set()
        corr_matrix: Figure = self.dataset.corr()
    
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > 0.8:
                    matrix = corr_matrix.columns[i]
                    corr_set.add(matrix)
    
        print(f"Number of Correrated features: {len(corr_set)}")
        print(f"List of Correrated Features: {list(corr_set)}")



    def feature_correration(self) -> pd.DataFrame:
        corr_matrix: Figure = self.dataset.corr()
        corr_matrix = corr_matrix.abs().unstack()
        corr_matrix = corr_matrix.sort_values(ascending = False)
        corr_matrix = corr_matrix[(corr_matrix >= 0.8) & (corr_matrix < 1)]
    
        corr_matrix: pd.DataFrame = pd.DataFrame(corr_matrix).reset_index()
        corr_matrix.columns = ["feature_1","feature_2", "correration"]
        return corr_matrix


# -------- Data Preprocess Class 
class DataPreprocess:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = X
        self.y = y


    def __repr__(self) -> tuple[str|Any, ...]:
        return self.__module__, type(self).__name__, hex(id(self))


    def __str__(self) -> dict[str, str]:
        info: list[str] = ["module", "name", "ObjectID"]
        return {item: value for item, value in zip(info, self.__repr__())} 


# ------ HardWare Config: Google's GPU or TPU  -> Docker 
class HardwareConfig:
    def Hardware_Accelerator(self) -> Text:
        device: Any = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using : {device}")



# ------ Feed_Forward_Net HyperParms Class
class NetHyperParms:
    # ---
    batch_size: int = 128
    validation_split: float = 0.3
    shuffle_dataset: bool = True
    random_seed: int = 0
    input_dim: int = 28*28
    hidden_dim: int = 150
    output_dim: int = 10
    learning_rate: int = 0.001
    epochs: int = 20
    # ...


# -------- FeedForward Neural Network : Architecture = [Tensor <3 ReLu's + 3 Linear's Layers>] 
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x: Tensor[Any]) -> Model:
        x: Tensor[Any] = self.flatten(x)
        logits: Model = self.linear_relu_stack(x) 
        return logits

    
    def train_loop(self, dataloader: Loader, model: Model, loss_fn: Loss, optimizer: Optimizor) -> Text|Any:
        size: int = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            pred: Predictor = model(X)
            loss: Loss = loss_fn(pred, y)
            # -.. network propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"Loss : {loss:>7f} [{current:>5d} / {size:>5d}]")


    def test_loop(self, dataloader: Loader, model: Model, loss_fn: Loss) -> Text:
        size: int = len(dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred: Predictor = model(X)
                test_loss: Loss = loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss: Loss = test_loss / size
        correct: Loss = correct / size
        print(f"Test Error : \n Accuracy: {(100*correct):>1f}%, Avg loss: {test_loss:>8f} \n")



# Driver Code: exec(1) if not __main_module__
if __name__ == "__main_module__":
    print("Hemllo")
    

