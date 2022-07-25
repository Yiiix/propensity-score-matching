import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, roc_curve, accuracy_score
from scipy.stats import ttest_ind_from_stats
import timeit


class psm(object):
    def __init__(self,k ,caliper = None  ):
        self.caliper = caliper
        self.k = int(k) 
    
    def fit(self,data,treatment,y,**kwargs):
        self.data = data
        self.treatment = treatment
        self.y = y
        self.data['index'] = self.data.index

      
        """about pre check variable"""
        check_data = data.drop(['index',treatment,y],axis=1)
        columns = check_data.columns
        column_name = []
        T_statistic = []
        p_value = []
        significance = []

        for i in columns:
            if data[i].dtype == 'float64' or data[i].dtype =='int64':
                mean1 = data.groupby(treatment)[i].mean()[0]
                mean2 = data.groupby(treatment)[i].mean()[1]

                std1 = data.groupby(treatment)[i].std()[0]
                std2 = data.groupby(treatment)[i].std()[1]

                nobs1 = data.value_counts(treatment)[0]
                nobs2 = data.value_counts(treatment)[1]

                mod_std1 = np.sqrt(nobs1/(nobs1-1))*std1
                mod_std2 = np.sqrt(nobs2/(nobs2-1))*std2
                (statistic, pvalue) = ttest_ind_from_stats(mean1=mean1, std1=mod_std1, nobs1=nobs1
                                                            , mean2=mean2, std2=mod_std2, nobs2=nobs2)
                column_name.append(i)
                T_statistic.append(statistic)
                p_value.append(pvalue)
                if pvalue <= 0.05:
                    significance.append('顯著')
                else :
                    significance.append('不顯著')
        
        self.pre_check_outcome = pd.DataFrame({'column_name':column_name,'T_statistic':T_statistic,'p_value':p_value,'significance':significance})


        """about propensity score calcuate"""
        lda = LDA()
        
        lda.fit(check_data,data[treatment])
        self.data["propensity"] = lda.predict_proba(check_data)[:,1]

        print("\n計算完成!")

    
        pred =lda.predict(check_data)
        conf_mat = confusion_matrix(self.data[treatment], pred)
        
        TP = conf_mat[1,1]
        TN = conf_mat[0,0]
        FP = conf_mat[0,1]
        FN = conf_mat[1,0]
        self.classfier_Accuracy = (TP+TN) / (TP+TN+FP+FN)
        self.classfier_Sensitivity = TP / (FN + TP)
        self.classfier_Precision = TP / (FP + TP)
        self.classfier_Auc = roc_auc_score(self.data[treatment],self.data["propensity"])


        """about match"""
        ratio = self.k
        data = self.data
        ignore_list = set()
        under_matched = set()
        unmatched = set()
        matched_controls = pd.DataFrame()


        controls = self.data[self.data[self.treatment] == 0]
        cases = self.data[self.data[self.treatment] == 1]

        caliper_r = self.caliper*(self.data.propensity.std())

        neigh = NearestNeighbors(radius=caliper_r)
        neigh.fit(controls[['propensity']].values)

        i = 1
        total_cases = cases.shape[0]
        start = timeit.default_timer()


    

        for index, case in cases.iterrows():
            
            pscore = case.propensity

            distances, indices = neigh.radius_neighbors([[pscore]])

            sample = controls.iloc[indices[0]]

            sample = sample[~sample['index'].isin(ignore_list)].copy()


            sample['DIST'] = abs(sample['propensity']- pscore)
            sample.sort_values(by='DIST', ascending=True, inplace=True)
            sample = sample.head(ratio).copy().reset_index(drop=True)

            if (sample.shape[0] < ratio and sample.shape[0] != 0):
                under_matched.add(case['index'])
            if (sample.shape[0] == 0):
                unmatched.add(case['index'])

            ignore_list.update(sample['index'])
            sample['MATCHED_CASE'] = case['index']
            matched_controls = matched_controls.append(sample, ignore_index=True)
            
            
            stop = timeit.default_timer()
            
            print("Current progress:", np.round(i/total_cases * 100, 2), "%")
            print("Current run time:", np.round((stop - start) / 60, 2), "min")
            
            i = i+1
        
        matched_controls = matched_controls.reset_index(drop=True)
        self.matched_data = self.data.iloc[matched_controls['index'],:].append(self.data.iloc[matched_controls['MATCHED_CASE'],:]).drop_duplicates(['index'])
        self.matched_controls = matched_controls
        self.unmatched = unmatched
        


        """about quality check"""

        df = self.matched_data
        T_statistic_after = []
        p_value_after = []
        significance_after = []
        for i in columns:
            if df[i].dtype == 'float64' or df[i].dtype == 'int64':
                mean1 = df.groupby(treatment)[i].mean()[0]
                mean2 = df.groupby(treatment)[i].mean()[1]

                std1 = df.groupby(treatment)[i].std()[0]
                std2 = df.groupby(treatment)[i].std()[1]

                nobs1 = df.value_counts(treatment)[0]
                nobs2 = df.value_counts(treatment)[1]

                mod_std1 = np.sqrt(nobs1/(nobs1-1))*std1
                mod_std2 = np.sqrt(nobs2/(nobs2-1))*std2
                (statistic, pvalue) = ttest_ind_from_stats(mean1=mean1, std1=mod_std1, nobs1=nobs1
                                                            , mean2=mean2, std2=mod_std2, nobs2=nobs2)
                T_statistic_after.append(statistic)
                p_value_after.append(pvalue) 
                if pvalue <= 0.05 :
                    significance_after.append('顯著')
                else :
                    significance_after.append('不顯著')
        
        self.quality_check_outcome = pd.DataFrame({'column_name':column_name,'T_statistic':T_statistic_after,'p_value':p_value_after,'significance':significance_after})
        self.data.drop(['index'],axis=1,inplace = True)


    def pre_check(self):
        print(self.pre_check_outcome)

    def model_performance(self):
        print("Accuracy",self.classfier_Accuracy )
        print("Sensitivity: ", self.classfier_Sensitivity)
        print("Precision: ",  self.classfier_Precision )
        print("Auc: ",  self.classfier_Auc )

    def overlap(self):
        sns.set()
        sns.kdeplot(x='propensity',shade=True,data=self.data[self.data[self.treatment]!=1])
        sns.kdeplot(x='propensity',shade=True,data=self.data[self.data[self.treatment]==1])
        plt.legend(['no','yes'])
    
    def match_outcome(self):
        print('總配對數',len(self.matched_data))
        print('實驗組配對數',len(self.matched_data[self.matched_data[self.treatment]==1]))
        

    def match_quality(self):
        print(self.quality_check_outcome)