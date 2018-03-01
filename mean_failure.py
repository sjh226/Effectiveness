import numpy as np
import pandas as pd
import sys
import pyodbc
import matplotlib.pyplot as plt


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
		try:
			df.loc[df['assetWellFlac'] == flac, 'days_since_fail'] = \
				pd.to_numeric(((df[df['assetWellFlac'] == flac]['surfaceFailureDate'] - \
				df[df['assetWellFlac'] == flac]['last_failure']) / \
				np.timedelta64(1, 'D')), errors='coerce')
		except:
			df.loc[df['assetWellFlac'] == flac, 'days_since_fail'] = np.nan
		return_df = return_df.append(\
						{'assetBU': df[df['assetWellFlac'] == flac]['assetBU'].unique()[0], \
						 'assetAPI': df[df['assetWellFlac'] == flac]['assetAPI'].unique()[0], \
						 'assetWellFlac': flac, \
                         'WellName': df[df['assetWellFlac'] == flac]['WellName'].unique()[0], \
						 'surfaceFailureType': fail_type, \
						 'mean_time_fail': df[df['assetWellFlac'] == flac]['days_since_fail'].mean()}, \
						 ignore_index=True)
	return return_df

def fail_loop(df):
	mean_fail_df = pd.DataFrame(columns=['assetBU', 'assetAPI', 'assetWellFlac', \
										 'surfaceFailureType', 'mean_time_fail', \
										 'fail_bin', 'WellName'])
	for failure in ['Compressor', 'Separator']:
		fail_df = mean_time(df[df['surfaceFailureType'] == failure], failure)
		fail_df.loc[:, 'fail_bin'] = pd.cut(fail_df[fail_df['mean_time_fail'].notnull()]['mean_time_fail'], \
											4, labels=['best', 'good', 'bad', 'worst'])
		fail_df.loc[:, 'fail_bin_equal'] = pd.qcut(fail_df[fail_df['mean_time_fail'].notnull()]['mean_time_fail'], \
												   4, labels=['best', 'good', 'bad', 'worst'])
		mean_fail_df = mean_fail_df.append(fail_df)
		plot_fails(fail_df[fail_df['mean_time_fail'].notnull()], failure)
		# print('Mean time failure for {}:'.format(failure))
		# print(fail_df[fail_df['mean_time_fail'].notnull()].mean())

	return mean_fail_df.sort_values(['surfaceFailureType', 'mean_time_fail'])

def plot_fails(df, fail_type):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.bar(np.arange(df.shape[0]), df.sort_values('mean_time_fail')['mean_time_fail'], 1)

	plt.xticks(rotation='vertical')
	plt.xlabel('Wells')
	plt.ylabel('Mean Time Between Failures')
	plt.title('Mean Fail Time for {}s'.format(fail_type))

	plt.savefig('images/mean_fail_{}.png'.format(fail_type.lower()))


if __name__ == '__main__':
	df = data_fetch()

	mean_fail_df = fail_loop(df)
    mean_fail_df.to_csv('data/mean_time_fail.csv')
    mean_fail_df = pd.read_csv('data/mean_time_fail.csv')
