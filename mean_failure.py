import numpy as np
import pandas as pd
import sys
import pyodbc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split


def data_fetch():
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=SQLDW-L48.BP.Com;'
									r'Database=EDW;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SELECT SFAD.assetBU
			  ,SFAD.assetAPI
			  ,SFAD.assetWellFlac
			  ,W.WellName
			  ,SFAD.createdDate
			  ,SFAD.modifiedDate
			  ,SFAD.assetName
			  ,SFAD.surfaceFailureDate
			  ,SFAD.surfaceFailureType
			  ,SFAD.compressionType
			  ,SFAD.surfaceFailureComponent
			  ,SFAD.surfaceFailureComponentOther
			  ,SFAD.surfaceFailureSubComponent
			  ,SFAD.surfaceFailureSubComponentOther
			  ,SFAD.surfaceFailureManufacturer
			  ,SFAD.surfaceFailureManufacturerOther
			  ,SFAD.surfaceFailureModel
			  ,SFAD.surfaceFailureModelOther
			  ,SFAD.surfaceFailureRootCause
			  ,SFAD.surfaceFailureRootCauseOther
			  ,SFAD.surfaceFailureDamages
		  FROM [EDW].[Enbase].[SurfaceFailureActionDetailed] AS SFAD
		  JOIN [OperationsDataMart].[Dimensions].[Wells] AS W
			ON W.WellFlac = SFAD.assetWellFlac
		  WHERE SFAD.assetBU = 'WAMSUTTER'
		  AND SFAD.deletedDate IS NULL
		  AND (SFAD.surfaceFailureType = 'Compressor'
			OR SFAD.surfaceFailureComponent = 'Choke Valve/Loop');
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None
		print('Dataframe is empty')

	return df.drop_duplicates()

def mean_time(df, fail_type):
	return_df = pd.DataFrame(columns=['assetBU', 'assetAPI', 'assetWellFlac', \
									  'WellName', 'surfaceFailureType', 'mean_time_fail'])

	df.loc[:, 'last_failure'] = np.nan
	df.loc[:, 'days_since_fail'] = np.nan
	df.sort_values('surfaceFailureDate', inplace=True)
	for flac in df[df['assetWellFlac'].notnull()]['assetWellFlac'].unique():
		df.loc[df['assetWellFlac'] == flac, 'last_failure'] = \
			df[df['assetWellFlac'] == flac]['surfaceFailureDate'].shift(1)
		df.loc[df['assetWellFlac'] == flac, 'next_failure'] = \
			df[df['assetWellFlac'] == flac]['surfaceFailureDate'].shift(-1)
		try:
			df.loc[df['assetWellFlac'] == flac, 'days_fixed'] = \
				pd.to_numeric(((df[df['assetWellFlac'] == flac]['next_failure'] - \
				df[df['assetWellFlac'] == flac]['surfaceFailureDate']) / \
				np.timedelta64(1, 'D')), errors='coerce')
		except:
			df.loc[df['assetWellFlac'] == flac, 'days_fixed'] = np.nan
		return_df = return_df.append(\
						{'assetBU': df[df['assetWellFlac'] == flac]['assetBU'].unique()[0], \
						 'assetAPI': df[df['assetWellFlac'] == flac]['assetAPI'].unique()[0], \
						 'assetWellFlac': flac, \
						 'WellName': df[df['assetWellFlac'] == flac]['WellName'].unique()[0], \
						 'surfaceFailureType': fail_type, \
						 'mean_time_fail': df[df['assetWellFlac'] == flac]['days_fixed'].mean()}, \
						 ignore_index=True)
	return return_df, df

def fail_loop(df):
	mean_fail_df = pd.DataFrame(columns=['assetBU', 'assetAPI', 'assetWellFlac', \
										 'surfaceFailureType', 'mean_time_fail', \
										 'fail_bin', 'WellName'])
	total_detail_df = pd.DataFrame(columns=df.columns)
	for failure in ['Compressor', 'Separator']:
		fail_df, detail_df = mean_time(df[df['surfaceFailureType'] == failure], failure)
		fail_df.loc[:, 'fail_bin'] = pd.cut(fail_df[fail_df['mean_time_fail'].notnull()]['mean_time_fail'], \
											4, labels=['worst', 'bad', 'good', 'best'])
		fail_df.loc[:, 'fail_bin_equal'] = pd.qcut(fail_df[fail_df['mean_time_fail'].notnull()]['mean_time_fail'], \
												   4, labels=['worst', 'bad', 'good', 'best'])
		mean_fail_df = mean_fail_df.append(fail_df)
		total_detail_df = total_detail_df.append(detail_df)
		# plot_fails(fail_df[fail_df['mean_time_fail'].notnull()], failure)
		# print('Mean time failure for {}:'.format(failure))
		# print(fail_df[fail_df['mean_time_fail'].notnull()].mean())

	return mean_fail_df.sort_values(['surfaceFailureType', 'mean_time_fail']), total_detail_df

def best_fix(fail_df, fix_df):
	best_fix_df = pd.DataFrame(columns=fix_df.columns)
	for failure in fail_df['surfaceFailureType'].unique():
		fail_time = fail_df[(fail_df['surfaceFailureType'] == failure) & \
							(fail_df['fail_bin_equal'] == 'best')]['mean_time_fail'].min()
		fail_fix_df = fix_df[(fix_df['surfaceFailureType'] == failure) & \
							 (fix_df['days_fixed'] >= fail_time)]
		best_fix_df = best_fix_df.append(fail_fix_df)
		# plot_root_cause(fail_fix_df, failure)

	return best_fix_df[['assetWellFlac', 'WellName', 'surfaceFailureDate', \
					    'surfaceFailureType', 'surfaceFailureComponent', \
					    'surfaceFailureSubComponent', 'surfaceFailureManufacturer', \
					    'surfaceFailureModel', 'surfaceFailureRootCause', \
					    'surfaceFailureRootCauseOther', 'surfaceFailureDamages', \
					    'days_fixed']].sort_values(['surfaceFailureType', 'days_fixed'])

def vectorize(df):
	vect = TfidfVectorizer(stop_words='english')
	X = vect.fit_transform(df[df['surfaceFailureDamages'].notnull()]['surfaceFailureDamages'].values)
	y = df[df['surfaceFailureDamages'].notnull()]['days_fixed'].values
	idf = vect.idf_

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=87)

	rf = RandomForestRegressor()
	rf.fit(X_train, y_train)
	print(rf.score(X_test, y_test))

def plot_fails(df, fail_type):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.bar(np.arange(df.shape[0]), df.sort_values('mean_time_fail')['mean_time_fail'], 1)

	plt.xticks(rotation='vertical')
	plt.xlabel('Wells')
	plt.ylabel('Mean Time Between Failures')
	plt.title('Mean Fail Time for {}s'.format(fail_type))

	plt.savefig('images/mean_fail_{}.png'.format(fail_type.lower()))

def plot_root_cause(df, fail_type):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	bar_heights = {}

	for cause in df[df['surfaceFailureRootCause'].notnull()]['surfaceFailureRootCause'].unique():
		bar_heights[cause] = df[df['surfaceFailureRootCause'] == cause].shape[0]

	ax.bar(bar_heights.keys(), bar_heights.values(), .9)

	plt.xticks(rotation='vertical')
	plt.xlabel('Surface Failure Root Cause')
	plt.ylabel('Failure Count')
	plt.title('Root Cause Counts for {}s'.format(fail_type))
	plt.tight_layout()

	plt.savefig('images/root_cause_{}.png'.format(fail_type.lower()))


if __name__ == '__main__':
	# df = data_fetch()
	# mean_fail_df, detail_df = fail_loop(df)
	# mean_fail_df.to_csv('data/mean_time_fail.csv')
	# detail_df.to_csv('data/detail_df.csv', encoding='utf-8')

	mean_fail_df = pd.read_csv('data/mean_time_fail.csv')
	detail_df = pd.read_csv('data/detail_df.csv')

	fix_df = best_fix(mean_fail_df, detail_df)
	vectorize(fix_df[fix_df['surfaceFailureType'] == 'Compressor'])

	# best_comp_df = fix_df[fix_df['surfaceFailureType'] == 'Compressor'].tail(10)
	# best_choke_df = fix_df[fix_df['surfaceFailureType'] == 'Separator'].tail(10)
	#
	# best_comp_df[['surfaceFailureType', 'surfaceFailureRootCause', 'surfaceFailureDamages']].to_csv('data/best_comp.csv', index=False)
	# best_choke_df[['surfaceFailureType', 'surfaceFailureRootCause', 'surfaceFailureDamages']].to_csv('data/best_choke.csv', index=False)
