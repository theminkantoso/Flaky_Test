import matplotlib.pyplot as plt
import matplotlib
import csv
import math

plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.rcParams["figure.autolayout"] = True

x = []
y1 = []
y2 = []
y3 = []
y4 = []
z1 = []
z2 = []
z3 = []
z4 = []

y11 = []
y12 = []
y13 = []
y14 = []
z11 = []
z12 = []
z13 = []
z14 = []
  
with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/eff-eff3.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    for row in lines:
        x.append(row[0])
        if (row[5] == '3' and row[6] == '0.5'):
            if(row[7] == '-'):
                y1.append(-0.1)
            else:
                y1.append(float(row[7]))
            z1.append(float(row[8]))
        elif (row[5] == '3' and row[6] == '0.95'):
            if(row[7] == '-'):
                y2.append(-0.1)
            else:
                y2.append(float(row[7]))
            z2.append(float(row[8]))
        elif (row[5] == '7' and row[6] == '0.5'):
            if(row[7] == '-'):
                y3.append(-0.1)
            else:
                y3.append(float(row[7]))
            z3.append(float(row[8]))
        elif (row[5] == '7' and row[6] == '0.95'):
            if(row[7] == '-'):
                y4.append(-0.1)
            else:
                y4.append(float(row[7]))
            z4.append(float(row[8]))

with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_naive_bayes.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    for row in lines:
        z11.append(float(row[6]))
        if(row[5] == '-'):
            y11.append(-0.1)
        else:
            y11.append(float(row[5]))

with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_random_forest.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    for row in lines:
        z12.append(float(row[6]))
        if(row[5] == '-'):
            y12.append(-0.1)
        else:
            y12.append(float(row[5]))

with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_svm.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    for row in lines:
        z13.append(float(row[6]))
        if(row[5] == '-'):
            y13.append(-0.1)
        else:
            y13.append(float(row[5]))

x=x[0:13]
plt.subplot(2,2,1)
plt.plot(x, z1, color = 'k', marker = '^', label = "KNN k = 3 sigma = 0.5")
plt.plot(x, z11, color = 'g', marker = '^', alpha = 0.2, linestyle="dashdot", label = "Naive Bayes n = 30")
plt.plot(x, z12, color = 'r', marker = 'v', alpha = 0.2, linestyle="dashdot", label = "Random Forest n = 20") 
plt.plot(x, z13, color = 'c', marker = '>', alpha = 0.2, linestyle="dashdot", label = "SVM n = 4") 
plt.xticks([])
plt.legend(bbox_to_anchor=(.45, -0.4), loc="lower center")
plt.grid()

plt.subplot(2,2,2)
plt.plot(x, z2, color = 'k', marker = '^', label = "KNN k = 3 sigma = 0.95")
plt.plot(x, z11, color = 'g', marker = '^', alpha = 0.2, linestyle="dashdot", label = "Naive Bayes n = 30")
plt.plot(x, z12, color = 'r', marker = 'v', alpha = 0.2, linestyle="dashdot", label = "Random Forest n = 20") 
plt.plot(x, z13, color = 'c', marker = '>', alpha = 0.2, linestyle="dashdot", label = "SVM n = 4")
plt.xticks([])
plt.legend(bbox_to_anchor=(.45, -0.4), loc="lower center")
plt.grid()

plt.subplot(2,2,3)
plt.plot(x, z3, color = 'k', marker = '^', label = "KNN k = 7 sigma = 0.5")
plt.plot(x, z11, color = 'g', marker = '^', alpha = 0.2, linestyle="dashdot", label = "Naive Bayes n = 30")
plt.plot(x, z12, color = 'r', marker = 'v', alpha = 0.2, linestyle="dashdot", label = "Random Forest n = 20") 
plt.plot(x, z13, color = 'c', marker = '>', alpha = 0.2, linestyle="dashdot", label = "SVM n = 4")
plt.xticks([])
plt.legend(bbox_to_anchor=(.45, -0.4), loc="lower center")
plt.grid()

plt.subplot(2,2,4)
plt.plot(x, z4, color = 'k', marker = '^', label = "KNN k = 7 sigma = 0.95")
plt.plot(x, z11, color = 'g', marker = '^', alpha = 0.2, linestyle="dashdot", label = "Naive Bayes n = 30")
plt.plot(x, z12, color = 'r', marker = 'v', alpha = 0.2, linestyle="dashdot", label = "Random Forest n = 20") 
plt.plot(x, z13, color = 'c', marker = '>', alpha = 0.2, linestyle="dashdot", label = "SVM n = 4")
plt.xticks([])
plt.legend(bbox_to_anchor=(.45, -0.4), loc="lower center")
plt.grid()

# plt.xticks(rotation = 60)
# plt.xlabel('dataset')
# plt.ylabel('Precision')
# plt.title('Compare with original paper', fontsize = 20)
plt.suptitle('Recall comparing different models', y=1)
# plt.grid()
# plt.legend(bbox_to_anchor=(.45, 1.15), loc="lower center")
plt.savefig('/content/gdrive/MyDrive/Colab Notebooks/FLAST/graphs/comparison/compare_models_recall.png', bbox_inches='tight')

# with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_naive_bayes.csv','r') as csvfile:
#     lines = csv.reader(csvfile, delimiter=',')
#     next(lines)
#     for row in lines:
#         x.append(row[0])
#         z1.append(float(row[6]))
#         if(row[5] == '-'):
#             y1.append(-0.1)
#         else:
#             y1.append(float(row[5]))

# with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_random_forest.csv','r') as csvfile:
#     lines = csv.reader(csvfile, delimiter=',')
#     next(lines)
#     for row in lines:
#         z2.append(float(row[6]))
#         if(row[5] == '-'):
#             y2.append(-0.1)
#         else:
#             y2.append(float(row[5]))

# with open('/content/gdrive/MyDrive/Colab Notebooks/FLAST/results/flast_svm.csv','r') as csvfile:
#     lines = csv.reader(csvfile, delimiter=',')
#     next(lines)
#     for row in lines:
#         z3.append(float(row[6]))
#         if(row[5] == '-'):
#             y3.append(-0.1)
#         else:
#             y3.append(float(row[5]))
            
# plt.plot(x, z1, color = 'g', marker = '^', label = "Naive Bayes n = 30")
# plt.plot(x, z2, color = 'r', marker = 'v', label = "Random Forest n = 20") 
# plt.plot(x, z3, color = 'c', marker = '>', label = "SVM n = 4", alpha = 0.5) 
# # plt.plot(x, z4, color = 'y', marker = '<', label = "k = 7 sigma = 0.95") 
# plt.xticks(rotation = 60)
# plt.xlabel('dataset')
# plt.ylabel('Recall')
# plt.title('Recall Each Project Testing Models', fontsize = 20)
# plt.grid()
# plt.legend(bbox_to_anchor=(.45, 1.15), loc="lower center")
# plt.savefig('/content/gdrive/MyDrive/Colab Notebooks/FLAST/graphs/testing_models_recall2.png', bbox_inches='tight')