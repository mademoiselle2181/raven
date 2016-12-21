def TIMELOCA(self,data):
  modifiedVariables=['ABBN(1)','ABBN(2)','ABBN(3)']
  branchDict1={}
  branchDict2={}
  branchDict3={}
  newValue=[]
  listDict=[]
  #for each branch, a dictionary contained the value of each variable (that is modified within the branch is defined)
  #es.
  #BRANCH 1
  branchDict1['TIM100']='1'
  branchDict1['ABBN(1)']='3.8E-004'
  branchDict1['ABBN(2)']='3.8E-004'
  branchDict1['ABBN(3)']='3.8E-004'
  #BRANCH 2
  branchDict2['TIM100']='1'
  branchDict2['ABBN(1)']='9.5E-004'
  branchDict2['ABBN(2)']='9.5E-004'
  branchDict2['ABBN(3)']='9.5E-004'
  #BRANCH 3
  branchDict3['TIM100']='1'
  branchDict3['ABBN(1)']='2.2E-004'
  branchDict3['ABBN(2)']='2.2E-004'
  branchDict3['ABBN(3)']='2.2E-004'

 #a dictionary is defined where each item is {probabilityBranch#:branchDitc#}
  self.dictTIMELOCA={'0.1':branchDict1,'0.4':branchDict2,'0.5':branchDict3}

  for var in modifiedVariables:
    newValue=[]
    for dictBranch in self.dictTIMELOCA.values():
      oldValue=str(data[var][0])
      newValue.append(dictBranch[var])
      probability= self.dictTIMELOCA.keys()

    branch={'name':var, 'type':'auxiliary','old_value': oldValue, 'new_value': ' '.join(newValue), 'associated_pb':' '.join(probability)}
    listDict.append(branch)

  return listDict



