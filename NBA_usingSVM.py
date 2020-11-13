from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import os
df = pd.read_csv('2012-18_teamBoxScore_diff_columns.csv')
df = df.dropna(how='any') 
target = df["outcome"]
target_names = ["loss", "win"]
df1 = df[['diff_teamEFG%','diff_opptEFG%', 'diff_teamTO%', 'diff_opptTO%', 'diff_OREB%',
          'diff_DREB%', 'diff_teamFTF', 'diff_opptFTF', 'outcome']]
data = df1.drop("outcome", axis=1)
feature_names = data.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

from sklearn.svm import SVC 
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)
print('Test Accuracy: %.3f' % model.score(X_test, y_test))

from sklearn.metrics import classification_report
predictions = model.predict(X_test)
print('Classification report:')
print(classification_report(y_test, predictions,target_names=target_names))
mil_bos = [[0.03, 0.003, 1.1, -2.6, -0.4, 3.1, 0.018, -0.044]]
mil_bos_prediction = model.predict(mil_bos)
mil_bos_prediction
lac_gsw = [[-0.021, -0.007, -0.3, -1.3, -0.8, -3.0, 0.06, 0.018]]
lac_gsw_prediction = model.predict(lac_gsw)
lac_gsw_prediction
gsw_sac = [[0.017, -0.021, 1.1, -2.5, 1.6, 3.4, 0.026, -0.01]]
gsw_sac_prediction = model.predict(gsw_sac)
gsw_sac_prediction

def predict_outcome(road_team_abbr, road_team_stats, home_team_stats):
    road_team_array = np.array(road_team_stats)
    home_team_array = np.array(home_team_stats)
    diffs = road_team_array - home_team_array
    diffs_l = [diffs]
    prediction = model.predict(diffs_l)
    if ((prediction==1.).all()==True):
        print('Prediction is a Win for ' + str(road_team_abbr))
    else:
        print('Prediction is a Loss for ' + str(road_team_abbr))

mil_stats = [.554, .506, 12.6, 11.5, 22.5, 81.0, .191, .162]
bos_stats = [.524, .503, 11.5, 14.1, 22.9, 77.9, .173, .206]
predict_outcome('MIL', mil_stats, bos_stats)
lac_stats = [.531, .507, 12.5, 10.0, 22.5, 75.1, .267, .218]
gsw_stats = [.552, .514, 12.8, 11.3, 23.3, 78.1, .207, .200]
predict_outcome('LAC', lac_stats, gsw_stats)
gsw_stats = [.552, .514, 12.8, 11.3, 23.3, 78.1, .207, .200]
sac_stats = [.535, .535, 11.7, 13.8, 21.7, 74.7, .181, .210]
predict_outcome('GSW', gsw_stats, lac_stats)
mil_stats = [.551, .504, 12.6, 11.4, 22.1, 80.9, .191, .163]
nyk_stats = [.495, .546, 11.9, 12.1, 24.3, 76.0, .205, .207]
predict_outcome('MIL', mil_stats, nyk_stats)
okc_stats = [.510, .508, 12.5, 15.1, 27.5, 78.0, .191, .195]
hou_stats = [.532, .531, 12.4, 12.7, 25.0, 75.1, .214, .226]
predict_outcome('OKC', okc_stats, hou_stats)
phi_stats = [.532, .501, 13.1, 10.7, 23.6, 79.1, .240, .212]
bos_stats = [.525, .502, 11.5, 13.9, 23.0, 78.0, .175, .207]
predict_outcome('PHI', phi_stats, bos_stats)
lal_stats = [.537, .510, 13.2, 12.3, 22.1, 76.0, .176, .177]
gsw_stats = [.552, .514, 12.8, 11.3, 23.3, 78.1, .207, .200]
predict_outcome('LAL', lal_stats, gsw_stats)
por_stats = [.519, .521, 12.6, 10.7, 25.6, 78.0, .208, .184]
uta_stats = [.525, .521, 13.9, 13.6, 21.6, 79.5, .218, .192]
predict_outcome('POR', por_stats, uta_stats)
was_stats = [.525, .541, 12.2, 13.7, 19.9, 72.0, .214, .221]
det_stats = [.495, .514, 12.2, 12.6, 25.4, 79.3, .202, .235]
predict_outcome('WAS', was_stats, det_stats)
pho_stats = [.512, .535, 14.1, 13.2, 20.2, 74.7, .176, .227]
orl_stats = [.506, .526, 12.1, 12.5, 20.0, 77.7, .159, .198]
predict_outcome('PHO', pho_stats, orl_stats)
ind_stats = [.527, .501, 13.2, 14.0, 22.1, 78.1, .175, .182]
atl_stats = [.509, .542, 15.3, 13.8, 22.9, 76.9, .201, .234]
predict_outcome('IND', ind_stats, atl_stats)
cho_stats = [.521, .527, 10.6, 12.7, 21.2, 76.5, .203, .188]
brk_stats = [.521, .523, 12.5, 11.7, 25.8, 76.1, .215, .212]
predict_outcome('CHO', cho_stats, brk_stats)
tor_stats = [.544, .505, 12.0, 12.7, 23.4, 75.7, .185, .184]
mia_stats = [.499, .503, 12.7, 12.4, 26.5, 77.1, .193, .217]
predict_outcome('TOR', tor_stats, mia_stats)
min_stats = [.506, .529, 11.4, 13.1, 24.1, 74.0, .214, .186]
chi_stats = [.499, .522, 13.7, 12.4, 16.9, 75.7, .179, .188]
predict_outcome('MIN', min_stats, chi_stats)
cle_stats = [.497, .551, 11.7, 11.0, 25.1, 76.6, .181, .176]
mem_stats = [.514, .512, 12.3, 14.5, 18.1, 77.2, .204, .232]
predict_outcome('CLE', cle_stats, mem_stats)
nop_stats = [.531, .529, 12.6, 11.4, 25.6, 76.8, .212, .195]
dal_stats = [.522, .528, 13.4, 12.9, 24.0, 77.9, .234, .182]
predict_outcome('NOP', nop_stats, dal_stats)
den_stats = [.521, .511, 11.9, 12.6, 27.1, 78.1, .186, .198]
sas_stats = [.531, .529, 11.0, 11.1, 22.1, 78.5, .199, .166]
predict_outcome('DEN', den_stats, sas_stats)
sac_stats = [.535, .535, 11.7, 13.8, 21.7, 74.7, .181, .210]
lac_stats = [.531, .507, 12.5, 10.0, 22.5, 75.1, .267, .218]
predict_outcome('SAC', sac_stats, lac_stats)