import numpy as np

graphSizeX = 400
graphSizeY = 400
numPop = 20000
lengthOfDay = 4
U = 0.6106590514037163
t = 0.12972837155715206
V = 0.007195437031323588
durationSickness = 14
number_processes = 4

time_lockdown = 21

factor_actual_cases = 100
U_before = 0.8005245610328524
t_before = 0.2449737295188037
V_before = 0.003966816879595484
U_after = 0.748347887761644
t_after = 0.012301912204795307
V_after = 0.003392599648981556





# factor_actual_cases = 10
# U_before = 0.754308140381451
# t_before = 0.07597361870513031
# V_before = 0.0022893019716647097
# U_after =  0.7636825727407794
# t_after = 0.05033500857411421
# V_after = 0.0024019548919742457








# Data from https://www.n-tv.de/infografik/Coronavirus-aktuelle-Zahlen-Daten-zur-Epidemie-in-Deutschland-Europa-und-der-Welt-article21604983.html
# from 08.03.2020 to 18.04.2020
number_sick_original = [1035, 1180, 1563, 1899, 2746, 3675, 4599, 5796, 7232, 9375, 12300, 15305, 19655, 22189, 21963, 26102, 30014, 34252, 37301, 42968, 48241, 52584, 48469, 50271, 53389, 57048, 60483, 63595, 65798, 66767, 67922, 59823, 60326, 61340, 61235, 61208, 59322, 59322, 61476, 59748, 57098, 55239, 54005 ]
number_immune_original = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2809, 2809, 2809, 2809, 5600, 5600, 5600, 5600, 13500, 16100, 18700, 21400, 23800, 26400, 28700, 30600, 33300, 46300, 50000, 53913, 57400, 60200, 64300, 63781, 68107, 73469, 78539, 82140]
population_germany = 80000000


number_sick = np.asarray(number_sick_original) * factor_actual_cases
number_immune = np.asarray(number_immune_original) * factor_actual_cases

number_sick = number_sick / float(population_germany) * numPop

n = number_sick[0] / float(numPop)





print('n = ', n)
print('factor_actual_cases = ', factor_actual_cases)
print('time_lockdown = ', time_lockdown)


