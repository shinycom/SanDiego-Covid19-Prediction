import matplotlib.pyplot as plt
import numpy  as np 
import pandas as pd

def plot_pic( rpt_new_cases, pred_cases, date, outputPic='foo2.png', title='' ) :
    assert len(rpt_new_cases) == len(date) and len(pred_cases) == len(date), 'Check data length, must be equal ! '
    xtickLoc=[]; xtickDat=[]; 
    #--- show data month by month ---
    # date : '3/7/2020', '4/7/2021', ..
    prevMonth = ''
    for i, d in enumerate(date) :
        mon = d.split('/')[0]
        if prevMonth != mon :
            prevMonth = mon
            xtickLoc.append(i)
            xtickDat.append(d)
    fig, ax = plt.subplots()
    ax.grid(True)
    l1 = ax.plot(rpt_new_cases, label = 'Daily reported new cases' )
    l2 = ax.plot(pred_cases,    label = 'Predicted new cases'    ) #
    #plt.legend( [l1, l2], ['Daily new cases' , 'Predicted new cases']) #, loc='upper left')
    plt.legend(loc='upper left', shadow=True)
    ax.set_xticks(xtickLoc)
    ax.set_xticklabels(xtickDat)
    plt.subplots_adjust(left=0.13, bottom=0.16, right=0.94, top=0.94)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    if title :
        fig.suptitle(title)
    else :
        fig.suptitle(' COVID19 Prediction for San Diego ')
    #ax.set_xlabel('Date')
    ax.set_ylabel('Number of new infectious people')

    plt.savefig(outputPic)

if __name__ == '__main__':
    df_new_cases  = pd.read_csv('data_cumulative_deaths.csv')
    #df_pred_cases = pd.read_csv('pred2.csv')
    df_pred_cases = pd.read_csv('trained.csv')
    daily_date = df_new_cases['Date']
    #print(df_new_cases.columns); print(df_pred_cases.columns);
    plot_pic(df_new_cases['Deaths'], df_pred_cases['infections'], df_new_cases['Date'])