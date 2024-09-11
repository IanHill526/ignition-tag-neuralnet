import mysql.connector
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class MakeTrainingData:
    def __init__(self, db,tableName='sqlt_data_1_2024_01',tagNames=None,tagPaths=None):
        self.db = db
        self.tableName = tableName
        self.tagNames = tagNames
        self.tagPaths = tagPaths #string label for tag eg 'inlet_flow'

    def queryData(self,tagPath,column):
        baseQry = """
        select 
            floatvalue
        from 
            %s
        where 
            %s.tagid = (select max(id) from sqlth_te where tagpath = '%s')""" % (self.tableName,self.tableName,tagPath)
        if column == 'plant_running':
            baseQry = baseQry.replace('floatvalue','intvalue')
        cursor = self.db.cursor()
        cursor.execute(baseQry)
        result = cursor.fetchall()
        if column == 'plant_running':
            print(result)
        df = pd.DataFrame(result, columns = [column])
        print(len(df))
        return df
        
    def dataFrame(self):
        data = {}
        df = self.queryData(self.tagPaths[0],self.tagNames[0])
        print(self.tagNames[0])
        for k in range(1,len(self.tagPaths)):
            print(self.tagNames[k])
            tempdf = self.queryData(self.tagPaths[k],self.tagNames[k])
            df = pd.concat([df,tempdf],axis=1)
        return df
    
    def makeCsv(self):
        df = self.dataFrame()
        df.to_csv('datatest.csv')

if __name__ == '__main__':
    # db = mysql.connector.connect(
    #     host="",
    #     user="",
    #     password="",
    #     database = ''
    # ) 
    #tableName = '' #mysql table name w/ historian data
    #labels = []#list of labels eg "flow_rate"
    #tagPaths = [] #list of ignition tags
    #d = MakeTrainingData(db,db_table_name,labels,tagPaths)
    #d.makeCsv()