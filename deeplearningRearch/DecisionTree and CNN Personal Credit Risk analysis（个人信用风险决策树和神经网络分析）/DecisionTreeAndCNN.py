import sys
sys.path.append('/home/aistudio/external-libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neural_network import  MLPClassifier


#
credit = pd.read_csv("credit.csv")
credit.head(10)
data = np.array(credit.values.tolist())
# 计算统计贷款期限和贷款申请额度
months_loan_duration_max=credit['months_loan_duration'].max()
months_loan_duration_min=credit['months_loan_duration'].min()
months_loan_duration_median=credit['months_loan_duration'].median()
print("统计贷款期限的最小值为")
print(months_loan_duration_min)
print("统计贷款期限的最大值为")
print(months_loan_duration_max)
print("统计贷款期限的中位数为")
print(months_loan_duration_median)
amount_max=credit['amount'].max()
amount_min=credit['amount'].min()
amount_median=credit['amount'].median()
print("贷款申请额度的最小值为")
print(amount_min)
print("贷款申请额度的最大值为")
print(amount_max)
print("贷款申请额度的中位数为")
print(amount_median)
# 统计违约客户的比例
def countDefault(credit):
    count = 0
    for c in credit['default']:
        if (c == 2):
            count += 1
    return count/1000
print("违约客户的比例为")
print(countDefault(credit))
# 将字符串变量用整数进行映射
cols = ['checking_balance','credit_history','purpose','savings_balance','employment_length','personal_status','other_debtors','property','installment_plan','housing','job','telephone','foreign_worker']
col_dicts = {}
col_dicts = {
    'checking_balance':{
        '1 - 200 DM': 2,
        '< 0 DM': 1,
        '> 200 DM': 3,
        'unknown': 0
    },
    'credit_history':{
        'critical': 0,
        'repaid': 1,
        'delayed': 2,
        'fully repaid': 3,
        'fully repaid this bank': 4
    },
    'purpose':{
        'radio/tv': 0,
        'car (new)': 1,
        'furniture': 2,
        'repairs': 3,
        'business': 4,
        'education': 5,
        'car (used)': 6,
        'domestic appliances': 7,
        'others': 8,
        'retraining': 9
    },
    'savings_balance':{
        'unknown': 0,
        '< 100 DM': 1,
        '101 - 500 DM': 2,
        '501 - 1000 DM': 3,
        '> 1000 DM': 4
    },
    'employment_length':{
        'unemployed': 0,
        '0 - 1 yrs': 1,
        '1 - 4 yrs': 2,
        '4 - 7 yrs': 3,
        '> 7 yrs' : 4
    },
    'personal_status':{
        'divorced male': 0,
        'single male': 1,
        'married male': 2,
        'female': 3
    },
    'other_debtors':{
        'none': 0,
        'guarantor': 1,
        'co-applicant': 2
    },
    'property':{
        'unknown/none': 0,
        'real estate': 1,
        'building society savings': 2,
        'other': 3
    },
    'installment_plan':{
        'none': 0,
        'bank': 1,
        'stores': 2
    },
    'housing':{
        'own': 0,
        'for free': 1,
        'rent': 2
    },
    'job':{
        'skilled employee': 0,
        'unskilled resident': 1,
        'mangement self-employed': 2,
        'unemployed non-resident': 3
    },
    'telephone':{
        'none': 0,
        'yes': 1
    },
    'foreign_worker':{
        'no': 0,
        'yes': 1
    }
}
for col in cols:
    credit[col] = credit[col].map(lambda x: x.strip())
    credit[col] = credit[col].map(col_dicts[col])

print("输出前五个样本经过编码后的结果")
print(credit.head(5))
# 划分训练数据和测试数据
x = credit.loc[:,'checking_balance':'foreign_worker']
y = credit['default']
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=1)
# 使用决策树模型进行训练
credit_model = DecisionTreeClassifier(min_samples_leaf=6,random_state=1)
credit_model.fit(x_train,y_train)
# 决策树模型性能评估
credit_pre = credit_model.predict(x_test)
print("模型分类结果为")
print(metrics.classification_report(y_test,credit_pre))
print(metrics.confusion_matrix(y_test,credit_pre))
print("模型的预测正确率为")
print(metrics.accuracy_score(y_test,credit_pre))
print("违约贷款的识别率为")
print(metrics.recall_score(y_test,credit_pre,pos_label=2))
# 模型性能提升，通过定义未违约和违约的代价权重来重新训练和评估模型
class_weights = {1:1,2:4}
credit_model_opt = DecisionTreeClassifier(max_depth=2,class_weight = class_weights)
credit_model_opt.fit(x_train,y_train)
credit_pre_opt = credit_model_opt.predict(x_test)
print("优化后的模型分类结果为")
print(metrics.classification_report(y_test,credit_pre_opt))
print(metrics.confusion_matrix(y_test,credit_pre_opt))
print("优化后模型的预测正确率为")
print(metrics.accuracy_score(y_test,credit_pre_opt))
print("优化后违约贷款的识别率为")
print(metrics.recall_score(y_test,credit_pre_opt,pos_label=2))
# 使用神经网络进行训练
clf = MLPClassifier()
clf.fit(x_train,y_train)

credit_pre_mlp = clf.predict(x_test)
print(metrics.accuracy_score(y_test,credit_pre_mlp))
print(metrics.classification_report(y_test,credit_pre_mlp))
# 性能评估
print("mlp模型的预测正确率为")
print(metrics.accuracy_score(y_test,credit_pre_mlp))
print("mlp模型违约贷款的识别率为")
print(metrics.recall_score(y_test,credit_pre_mlp,pos_label=2))