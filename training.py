'''
Script to run training using SEIR model.

Sample usage: `python training.py `
'''

import pandas as pd
import numpy  as np 
import datetime
from run_model import main
from plot_data import plot_pic
import json


def loadReportedData() : 
    ''' Reported new infected cases and new deaths started from 3/7/2020 -- 4/20/2021
    '''
    df_daily_cases = pd.read_csv('data_cumulative_cases.csv')
    df_daily_death = pd.read_csv('data_cumulative_deaths.csv')
    df_daily_cases['daily'] = df_daily_cases['Cumulative total'].diff().fillna(0)
    df_daily_death['daily'] = df_daily_death['Deaths'].diff().fillna(0)
    assert len(df_daily_cases['daily']) == len(df_daily_death['daily']), 'reported data should be alligned.'
    return (df_daily_cases['daily'], df_daily_death['daily'], df_daily_cases['Date'])

def cost( pred , rlt, weight=False) : 
    assert len(pred)==len(rlt), 'prediction and true values must have the same length, pred={}, rlt={} '.format(len(pred),len(rlt))
    dif   = sum ( ( pred - rlt ) ** 2 )
    if weight : 
    	score = dif*weight
    else :
    	score = dif
    return np.sqrt(score)
'''
def gridSearch() : 
    
	grid = [
            "INITIAL_R_0" ,
            ( 0.71.0, 1.1 , 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.1, 3.3, 3.6, 3.9 )
            
        ],
        [
            "LOCKDOWN_R_0" : 
            0.9500000000000001
        ],
        [
            "INFLECTION_DAY",
            "2020-03-22"
        ],
        [
            "RATE_OF_INFLECTION",
            0.4595564472923688
        ],
        [
            "LOCKDOWN_FATIGUE",
            1.0
        ],
        [
            "DAILY_IMPORTS",
            120.0
        ],
        [
            "MORTALITY_RATE",
            0.008
        ],
        [
            "REOPEN_DATE",
            "2020-05-23"
        ],
        [
            "REOPEN_SHIFT_DAYS",
            -20.0
        ],
        [
            "REOPEN_R",
            1.2
        ],
        [
            "REOPEN_INFLECTION",
            0.275
        ],
        [
            "POST_REOPEN_EQUILIBRIUM_R",
            0.959645253335419
        ],
        [
            "FALL_R_MULTIPLIER",
            1.0004080109154216
        ]

	
	return 
'''
def fixed_args() : 
    class Args :
        pass 
    args = Args()
    args.best_params_type = 'mean'
    args.best_params_dir  = './'
    args.country   = 'USA'
    args.region    = 'CA'
    args.subregion = 'San_Diego'
    args.skip_hospitalizations = True 
    args.quarantine_perc = 0 
    args.quarantine_effectiveness = -1
    args.verbose  =  True 
    args.simulation_start_date = '2020-02-28' #'2020-02-28'
    args.simulation_end_date   = '2021-04-20' #'2021-04-01'
    args.set_param = 0
    args.change_param = 0
    args.save_csv = False #'trained.csv' #0
    return args

def get_params():
    #args.change_param = ('INITIAL_R_0')
    #region_params = {'population' : 332000000}
    params_dict = {
            'INITIAL_R_0' : 3.24,
            'LOCKDOWN_R_0' : 0.9,
            'INFLECTION_DAY' : '2020-02-28', # first reported on 2020-03-07
            'RATE_OF_INFLECTION' : 0.25,
            'LOCKDOWN_FATIGUE' : 1.03,
            'DAILY_IMPORTS' : 200,
            'MORTALITY_RATE' : 0.195,
            'REOPEN_DATE' : '2020-03-01', #05-23
            'REOPEN_SHIFT_DAYS': 0,
            'REOPEN_R' : 1.2,
            'REOPEN_INFLECTION' : 0.3,
            'POST_REOPEN_EQUILIBRIUM_R' : 7.,
            'FALL_R_MULTIPLIER' : 1.001,
        }
    return params_dict

def train_model() : 
    args = fixed_args()
    #print(dates[:5]);print(infections[:5]);print(deaths[:5]);
    daily_new_cases, daily_new_death, daily_date = loadReportedData()
    assert sum(daily_new_cases < 0) == 0 and sum(daily_new_death < 0) == 0
    tune_params = {
                'INITIAL_R_0'        : [ r/10  for r in range(1, 50, 1) ],
                'LOCKDOWN_R_0'       : [ r/10  for r in range(1, 15, 1) ], #15
                #'RATE_OF_INFLECTION' : [ r/100 for r in range(1, 99,1) ], #0.25,
                'LOCKDOWN_FATIGUE'   : [ r/100 for r in range(50, 130) ],
                #'DAILY_IMPORTS'      : [ r     for r in range(10, 200, 10) ],
                'MORTALITY_RATE'     : [ r/200 for r in range(1, 40) ],
                'REOPEN_R'           : [ r/10  for r in range(1, 100, 1) ],
                'REOPEN_DATE'        : ['2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', 
                                        '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',
                                       ],
                #'REOPEN_INFLECTION'  : [ r/10  for r in range(10, 70) ], # has no effects on score
                #'REOPEN_SHIFT_DAYS'  : [ r     for r in range(0, 300) ], # little effect on score
                'POST_REOPEN_EQUILIBRIUM_R' : [ r/10  for r in range(2, 80) ],
                #'FALL_R_MULTIPLIER'         : [ r/10  for r in range(2, 80) ], # no effect on score


    }

    params_dict = get_params()
    bestscore = -1 ; setvalue = False ; 
    for param, valueRange in tune_params.items() : 
        for value in valueRange : 
            params_dict[param] = value
            args.set_param = tuple(params_dict.items())
            dates, infections, hospitalizations, deaths = main(args)
            newscore = cost(daily_new_cases, infections[8:])
            if bestscore < 0 or bestscore >= newscore :
                bestscore  = newscore
                bestvalue  = value
                setvalue   = True
                print('best new score for param')
                print(param)
                print(bestvalue)
            print(newscore)
            #plot_pic(daily_new_cases, infections[8:], daily_date, outputPic='foo3.png', title=param+str(bestvalue))
            #input('----- Checking output pic here ')

        if setvalue : 
            params_dict[param] = bestvalue
            print('set up best value for param ...') 
            print(param)
            print(bestvalue)  
        else :
            print('checking here')
            assert 0 
        setvalue   = False
    print('print and save best parameters ...')
    print(params_dict)
    print(bestscore)
    plot_pic(daily_new_cases, infections[8:], daily_date, outputPic='foo3.png')
    f = open('test.txt', 'wt')
    f.write(str(params_dict))
    f.close()
    if False :
        dates_str = np.array(list(map(str, dates)))
        combined_arr = np.vstack((dates_str, infections, hospitalizations, deaths)).T
        headers = 'dates,infections,hospitalizations,Deaths,mean_r_t'
        np.savetxt('pred_data.csv', combined_arr, '%s', delimiter=',', header=headers)

if __name__ == '__main__':
    train_model()

