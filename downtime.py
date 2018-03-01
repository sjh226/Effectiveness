import cx_Oracle
import pyodbc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr


def down_wells():
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=SQLDW-L48.BP.Com;'
									r'Database=OperationsDataMart;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SET NOCOUNT ON;
		DROP TABLE IF EXISTS #DaysDown;

		WITH theData
		AS (SELECT  WellName
					,WellFlac
					,API
					,Datekey
					,Asset
					,Area
					,Oil
					,Gas
					,Water
					,LinePressure
					,TubingPressure
					,CasingPressure
					,LastChokeStatus
					,LastChokeAction
					,LastChokeComment
					,CASE WHEN (avg(Gas) over(order by Datekey rows between 6 preceding and current row)) = 0
						THEN 'Flag'
						ELSE ''
						END AS WeeklyAvg
					,ROW_NUMBER() OVER (PARTITION BY WellName ORDER BY DateKey DESC) AS RK
		  FROM [OperationsDataMart].[Reporting].[AllData]
		  Where Gas = 0)

		SELECT D.WellName,
			   D.API,
			   D.WellFlac,
			   DATEADD(d, D.RK, D.DateKey) AS [Grouper],
			   COUNT(D.DateKey) AS DaysDown,
			   MAX(D.DateKey) AS LastDayDown,
			   MIN(D.DateKey) AS [FirstDayDown],
			   ROW_NUMBER() OVER (PARTITION BY D.WellName  ORDER BY DATEADD(d, D.RK, D.DateKey) DESC) AS descRK
		INTO #DaysDown
		FROM theData D
		GROUP BY D.WellName, D.API, D.WellFlac,
			   DATEADD(d, D.RK, D.DateKey);

		SELECT	DD.WellName
				,DD.WellFlac
				,DD.Grouper
				,DD.DaysDown
				,DD.LastDayDown
				,SFAD.surfaceFailureType
				,SFAD.surfaceFailureComponent
				,SFAD.surfaceFailureSubComponent
				,DD.FirstDayDown
				,DD.descRK
		FROM #DaysDown AS DD
		JOIN [EDW].[Enbase].[SurfaceFailureActionDetailed] AS SFAD
		  ON SFAD.assetWellFlac = DD.WellFlac
		  AND SFAD.surfaceFailureDate = DD.LastDayDown
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.columns = [col.lower() for col in df.columns]
		# df.columns = map(camel_convert, df.columns)
	except:
		df = None
		print('Dataframe is empty')

	return df.drop_duplicates()

def camel_convert(name):
	s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
	return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def average_down(df):
	for f_type in df['surfacefailuretype'].unique():
		type_df = df[df['surfacefailuretype'] == f_type]
		total = type_df.shape[0]
		avg_time = type_df['daysdown'].mean()
		# print('For failure: {}'.format(f_type))
		# print('Total of {} failures.'.format(total))
		# print('Average time for failure is {} days.'.format(avg_time))
		# print('------------------------------------------')
		downtime_histo(type_df, avg_time)

def downtime_histo(df, average):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	sf_type = df['surfacefailuretype'].unique()[0]

	upper_iqr = df['daysdown'].mean() + iqr(df['daysdown'], rng=(50, 75))
	lower_iqr = df['daysdown'].mean() - iqr(df['daysdown'], rng=(25, 50))
	result_df = df[df['daysdown'] <= upper_iqr]

	ax.hist(result_df['daysdown'], 20, width=1, label='Full Dataset')
	ax.axvline(average, color='red', linestyle='--', label='Average Downtime')

	plt.title('Downtime for {} Failures'.format(sf_type))
	plt.xlabel('Days Down From Failure')
	plt.ylabel('Count')
	plt.legend()

	plt.tight_layout()
	plt.savefig('images/downtime_{}.png'.format(sf_type.lower().replace(' ', '_').replace('/', '_')))


if __name__ == '__main__':
	df = down_wells()
	# df.to_csv('data/down_failures.csv')
	# df = pd.read_csv('data/down_failures.csv')

	# average_down(df)
