import numpy as np
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from itertools import cycle
from matplotlib import pyplot as plt
from prettytable import PrettyTable

warnings.filterwarnings("ignore")
No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'BFGO-GC-CSA-\nAXAI', 'BMO-GC-CSA-\nAXAI', 'MOA-GC-CSA-\nAXAI', 'POA-GC-CSA-\nAXAI', 'RATPO-GC-CSA-\nAXAI']
    Algorithm1 = ['TERMS', 'BFGO-GC-CSA-AXAI', 'BMO-GC-CSA-AXAI', 'MOA-GC-CSA-AXAI', 'POA-GC-CSA-AXAI', 'RATPO-GC-CSA-AXAI']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm1[0], Terms)
        for j in range(len(Algorithm1) - 1):
            Table.add_column(Algorithm1[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Confusion():
    for n in range(No_of_Dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        class_1 = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
        class_2 = ['MildDemented', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
        class_3 = ['MildDemented', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

        classes = [class_1, class_2, class_3]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.canvas.manager.set_window_title('Confusion Matrix')

        confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes[n])
        cm_display.plot(ax=ax)

        for labels in cm_display.text_.ravel():
            labels.set_fontsize(16)

        # Move plot upward
        fig.subplots_adjust(top=0.90, bottom=0.35)

        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix", y=1.05)  # move title a bit higher
        plt.ylabel('Actual', fontname="Arial", fontsize=18, fontweight='bold', color='k')
        plt.xlabel('Predicted', fontname="Arial", fontsize=18, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", rotation=50, fontsize=12, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", rotation=50, fontsize=12, fontweight='bold', color='k')

        plt.savefig(path, bbox_inches="tight")
        plt.show()


def Plot_ROC_Curve():
    cls = ['AlzheimerNet', 'Efficient-Net \nSqueeze \nAttention Block', 'CNN with self-\nattention', 'GC-CSA', 'RATPO-GC-\nCSA-AXAI']
    for a in range(No_of_Dataset):  # For 5 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", fontsize=12)
        path = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'BFGO-GC-CSA-\nAXAI', 'BMO-GC-CSA-\nAXAI', 'MOA-GC-CSA-\nAXAI', 'POA-GC-CSA-\nAXAI', 'RATPO-GC-CSA-\nAXAI']
    Classifier = ['BatchSize', 'AlzheimerNet', 'Efficient-Net \nSqueeze \nAttention Block', 'CNN with self-\nattention', 'GC-CSA', 'RATPO-GC-\nCSA-AXAI']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 3, 5, 9]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['4', '8', '16', '32', '48']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batchsize)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 3, 5, 9, 7, 12]
    bar_width = 0.15
    kfold = [1, 2, 3, 4, 5]
    Algorithm = ['BFGO-GC-CSA-\nAXAI', 'BMO-GC-CSA-\nAXAI', 'MOA-GC-CSA-\nAXAI', 'POA-GC-CSA-\nAXAI', 'RATPO-GC-CSA-\nAXAI']
    Classifier = ['AlzheimerNet', 'Efficient-Net \nSqueeze \nAttention Block', 'CNN with self-\nattention', 'GC-CSA', 'RATPO-GC-\nCSA-AXAI']

    Act_Fun = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid']

    colors = ['#eb0000', '#ebeb00', '#009a60', '#79eba2', '#0065a6']
    line_styles = ['-', '--', '-.', ':', 'dotted']

    for i in range(No_of_Dataset):

        for m in range(len(Graph_Terms)):

            Graph = eval[i, :, :5, Graph_Terms[m] + 4]

            plt.figure(figsize=(8, 5))

            for j in range(Graph.shape[1]):
                plt.plot(
                    Act_Fun,
                    Graph[:, j],
                    label=Algorithm[j],
                    color=colors[j % len(colors)],
                    linestyle=line_styles[j % len(line_styles)],
                    linewidth=2.5,
                    marker='o',
                    markersize=7
                )

            plt.title(f"{Terms[Graph_Terms[m]]} (Dataset {i + 1})", fontsize=14, fontweight='bold')
            plt.xlabel("Activation Function", fontsize=12, fontweight='bold')
            plt.ylabel(Terms[Graph_Terms[m]], fontsize=12, fontweight='bold')

            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
                       fontsize=9, frameon=True, shadow=True)

            plt.tight_layout()

            path = "./Results/Dataset_%s_%s_line_Alg.png" % (i + 1, Terms[Graph_Terms[m]])
            plt.savefig(path)
            # plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            plt.figure(figsize=(8, 5))

            for j in range(Graph.shape[1]):
                plt.plot(
                    Act_Fun,
                    Graph[:, j],
                    label=Classifier[j],
                    color=colors[j % len(colors)],
                    linestyle=line_styles[j % len(line_styles)],
                    linewidth=2.5,
                    marker='o',
                    markersize=7
                )

            plt.title(f"{Terms[Graph_Terms[m]]} (Dataset {i + 1})", fontsize=14, fontweight='bold')
            plt.xlabel("Activation Function", fontsize=12, fontweight='bold')
            plt.ylabel(Terms[Graph_Terms[m]], fontsize=12, fontweight='bold')

            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
                       fontsize=9, frameon=True, shadow=True)

            plt.tight_layout()

            path = "./Results/Dataset_%s_%s_line_Mod.png" % (i + 1, Terms[Graph_Terms[m]])
            plt.savefig(path)
            # plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def Plot_Proposed_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)

    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Classifier = ['AlzheimerNet', 'Efficient-Net \nSqueeze \nAttention Block', 'CNN with self-\nattention', 'GC-CSA', 'RATPO-GC-\nCSA-AXAI']
    Graph_Terms = [0, 3, 5, 7, 9, 12,]

    colors = [
        '#b388eb',  # Light Purple
        '#8ac926',  # Green
        '#deeb34',  # Lime Yellow
        '#ffa600',  # Orange
        '#ff6361',  # Coral Red
        '#58508d',  # Indigo
        '#bc5090',  # Magenta
        # '#003f5c',  # Deep Blue
        # '#7a5195',  # Violet
        # '#ef5675'  # Pinkish Red
    ]
    Epoch = [10, 20, 30,
             40, 50, 60,
             70]

    bar_width = 0.80
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[i, :, 4, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title("Classifier Comparison")

            X = np.arange(len(Epoch))

            plt.bar(X, Graph, color=colors, edgecolor='w',
                    linewidth=2, width=bar_width)

            plt.xlabel('No of Epoch', fontname="Arial", fontsize=18,
                       fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=18, fontname="Arial",
                       fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')

            # --- Remove x-axis completely ---
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(True)

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)

            dot_markers = [
                plt.Line2D([2], [2], marker='s', color='w',
                           markerfacecolor=colors[a], markersize=10)
                for a in range(len(Epoch))
            ]
            plt.legend(dot_markers, Epoch,
                       loc='upper center', bbox_to_anchor=(0.5, 1.10),
                       fontsize=10, frameon=False, ncol=len(Epoch))

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            path = "./Results/Dataset_%s_%s_Proposed_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    # plotConvResults()
    # Confusion()
    # Plot_ROC_Curve()
    Plots_Results()
    # Plot_Proposed_Results()
    # Table()
