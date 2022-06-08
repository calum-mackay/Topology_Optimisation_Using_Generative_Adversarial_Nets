#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil

from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

from abaqus import *
from abaqusConstants import *

session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry = COORDINATE)
# ****************************************************************************
# will need a range of around 32 in each case for final run
import numpy as np

L = 10
H = 5
F = -1000000

inputs = np.empty((0,2), float)

startval = 0

for v in range(startval,(startval+1)):
    for l in range(0,32):
        inputs = np.vstack((inputs, [0.2+v*0.00625, 0.1+l*0.1548]))
        
# ****************************************************************************
# for loop contaning repeating optimsation task

for i in range(0,(len(inputs))):
    
    Vf = inputs[i][0]
    # V = 0.3
    Fy = inputs[i][1]
    # F_y = 3.0
    

    # set up grid: part > create part

    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)


    mdb.models['Model-1'].sketches['__profile__'].Spot(point=(0.0, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].Spot(point=(0.0, H))
    mdb.models['Model-1'].sketches['__profile__'].Spot(point=(L, H))
    mdb.models['Model-1'].sketches['__profile__'].Spot(point=(L, Fy))
    mdb.models['Model-1'].sketches['__profile__'].Spot(point=(L, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(
        0.0, H))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((0.0, H/2))
    mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((0.0, H/2), 
        ))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, H), point2=(
        L, H))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, H))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, H), 
        ))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((0.0, H/2))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, H))
    mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
        addUndoState=False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((0.0, H/2), )
        , entity2=mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((
        L/2, H), ))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(L, H), point2=(
        L, Fy))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, (Fy + 0.5*(H - Fy))))
    mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, (Fy + 0.5*(H - Fy))), 
        ))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, H))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, (Fy + 0.5*(H - Fy))))
    mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
        addUndoState=False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, H), )
        , entity2=mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((
        L, (Fy + 0.5*(H - Fy))), ))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(L, Fy), point2=(
        L, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, Fy/2))
    mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, Fy/2), 
        ))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, (Fy + 0.5*(H - Fy))))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, Fy/2))
    mdb.models['Model-1'].sketches['__profile__'].ParallelConstraint(addUndoState=
        False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, (Fy + 0.5*(H - Fy))), 
        ), entity2=mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((
        L, Fy/2), ))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(L, 0.0), point2=(
        0.0, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, 0.0), 
        ))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, Fy/2))
    mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L/2, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
        addUndoState=False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((L, Fy/2), 
        ), entity2=mdb.models['Model-1'].sketches['__profile__'].geometry.findAt((
        L/2, 0.0), ))
    mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='Plate', type=
        DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['Plate'].BaseShell(sketch=
        mdb.models['Model-1'].sketches['__profile__'])


    del mdb.models['Model-1'].sketches['__profile__']


    # set material as steel : property > 1st box

    mdb.models['Model-1'].Material(name='Steel')
    mdb.models['Model-1'].materials['Steel'].Elastic(table=((207000000000.0, 0.3), 
        ))

    # set plane stress/strain conditions and thickness 1 > 2nd box

    mdb.models['Model-1'].HomogeneousSolidSection(material='Steel', name=
        'Section-1', thickness=1.0)


    # assign section properties : property > 3rd box

    mdb.models['Model-1'].parts['Plate'].Set(faces=
        mdb.models['Model-1'].parts['Plate'].faces.findAt(((L/2, H/2, 0.0), (0.0, 
        0.0, 1.0)), ), name='Set-1')
    mdb.models['Model-1'].parts['Plate'].SectionAssignment(offset=0.0, offsetField=
        '', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Plate'].sets['Set-1'], sectionName='Section-1'
        , thicknessAssignment=FROM_SECTION)


    # define the assembly > instances > create

    mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
    mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Plate-1', part=
        mdb.models['Model-1'].parts['Plate'])


    # linear static case: step > 1st box > set minimum step to 0.1

    mdb.models['Model-1'].StaticStep(initialInc=0.1, name='Step-1', previous=
        'Initial')


    # set end BCs as built-in: load > 2nd box > select edges > ENCASTBC

    mdb.models['Model-1'].rootAssembly.Set(edges=
        mdb.models['Model-1'].rootAssembly.instances['Plate-1'].edges.findAt(((0.0, 
        H/2, 0.0), )), name='Set-1')
    mdb.models['Model-1'].EncastreBC(createStepName='Initial', localCsys=None, 
        name='BC-1', region=mdb.models['Model-1'].rootAssembly.sets['Set-1'])


    # apply load based on additional point from sketch

    mdb.models['Model-1'].rootAssembly.Set(name='Set-2', vertices=
        mdb.models['Model-1'].rootAssembly.instances['Plate-1'].vertices.findAt(((
        L, Fy, 0.0), )))
    mdb.models['Model-1'].ConcentratedForce(cf2=F, createStepName=
        'Step-1', distributionType=UNIFORM, field='', localCsys=None, name='Load-1'
        , region=mdb.models['Model-1'].rootAssembly.sets['Set-2'])



    # *****
    # create the mesh: mesh > switch to part > highlight part > ok

    mdb.models['Model-1'].parts['Plate'].setElementType(elemTypes=(ElemType(
        elemCode=CPS4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
        elemCode=CPS3, elemLibrary=STANDARD)), regions=(
        mdb.models['Model-1'].parts['Plate'].faces.findAt((L, H/2, 0.0), (0.0, 
        0.0, 1.0)), ), )


    # mesh settings

    mdb.models['Model-1'].parts['Plate'].setMeshControls(algorithm=MEDIAL_AXIS, 
        elemShape=QUAD, regions=mdb.models['Model-1'].parts['Plate'].faces.findAt((
        (L/2, H/2, 0.0), (0.0, 0.0, 1.0)), ))


    # generate mesh size 0.1 on surface

    mesh_step = 0.1

    mdb.models['Model-1'].parts['Plate'].seedPart(deviationFactor=0.1, 
        minSizeFactor=0.1, size=mesh_step)
    mdb.models['Model-1'].parts['Plate'].generateMesh()


    # create a job to run the static load test

    mdb.models['Model-1'].rootAssembly.regenerate()
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name='Static-Run-'+str(i+1), nodalOutputPrecision=SINGLE
        , numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
        ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)


    # settings input complete run = >

    mdb.jobs['Static-Run-'+str(i+1)].submit(consistencyChecking=OFF)

    mdb.jobs['Static-Run-'+str(i+1)].waitForCompletion()


    # *************************************************************************************************
    # save static load results 
    beam_viewport = session.Viewport(name='Beam Results Viewport')
    # change the string to an iterative string here:
    beam_Odb_Path = 'Static-Run-'+str(i+1)+'.odb'
    an_odb_object = session.openOdb(name=beam_Odb_Path)
    beam_viewport.setValues(displayedObject=an_odb_object)

    session.viewports['Viewport: 1'].setValues(displayedObject=an_odb_object)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1)
    report_name_and_path = 'ReportStressResults'+str(i+1)+'.csv'
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)

    # to save the stress values
    session.writeFieldReport(fileName=report_name_and_path, append=OFF,sortItem='Element Label', 
                             odb=an_odb_object, step=0, frame=1,outputPosition=NODAL, 
                             variable=(('S', INTEGRATION_POINT, ((INVARIANT, 'Mises'), )), ))

    # *************************************************************************************************


    # END OF STATIC LOAD TEST!



    # Set up a topology optimization task

    mdb.models['Model-1'].rootAssembly.Set(faces=
        mdb.models['Model-1'].rootAssembly.instances['Plate-1'].faces.findAt(((L/2, 
        H/2, 0.0), (0.0, 0.0, 1.0)), ), name='Set-3')
    mdb.models['Model-1'].TopologyTask(name='Topology_Optmizer', region=
        mdb.models['Model-1'].rootAssembly.sets['Set-3'])



    # setup strain energy and volume fraction variables

    mdb.models['Model-1'].optimizationTasks['Topology_Optmizer'].SingleTermDesignResponse(
        drivingRegion=None, identifier='STRAIN_ENERGY', name='SE', operation=SUM, 
        region=MODEL, stepOptions=())
    mdb.models['Model-1'].optimizationTasks['Topology_Optmizer'].SingleTermDesignResponse(
        drivingRegion=None, identifier='VOLUME', name='VF', operation=SUM, region=
        MODEL, stepOptions=())


    # set up objective function to minimise strain energy

    mdb.models['Model-1'].optimizationTasks['Topology_Optmizer'].ObjectiveFunction(
        name='Objective-1', objectives=((OFF, 'SE', 1.0, 0.0, ''), ))


    # set up volume fraction constraint (VARIABLE)

    mdb.models['Model-1'].optimizationTasks['Topology_Optmizer'].OptimizationConstraint(
        designResponse='VF', name='Opt-Constraint-1', restrictionMethod=
        RELATIVE_LESS_THAN_EQUAL, restrictionValue = Vf)


    # Optimisation JOB: Set every cylce and 30 iterations

    mdb.OptimizationProcess(dataSaveFrequency=OPT_DATASAVE_EVERY_CYCLE, 
        description='', maxDesignCycle=30, model='Model-1', name='Opt-Process-'+str(i+1), 
        odbMergeFrequency=2, prototypeJob='Opt-Process-'+str(i+1)+'-Job', task=
        'Topology_Optmizer')
    mdb.optimizationProcesses['Opt-Process-'+str(i+1)].Job(atTime=None, 
        getMemoryFromAnalysis=True, memory=90, memoryUnits=PERCENTAGE, model=
        'Model-1', multiprocessingMode=DEFAULT, name='Opt-Process-'+str(i+1)+'-Job', numCpus=1
        , numGPUs=0, queue=None, waitHours=0, waitMinutes=0)



    # submit job and wait

    mdb.optimizationProcesses['Opt-Process-'+str(i+1)].submit()

    mdb.optimizationProcesses['Opt-Process-'+str(i+1)].waitForCompletion()





    # combine results!! must change the file location here as well for secondary use


# In[ ]:





# In[ ]:




    mdb.CombineOptResults(analysisFieldVariables=ALL, models=ALL, optIter = LAST, steps=('STEP-1', ), optResultLocation='H:\ABAQUS\FINALRUN-1\Opt-Process-'+str(i+1)
                         ,includeResultsFrom = LAST)



    # need to chaneg file odb path to:  'H:\ABAQUS\AutomationTesting\Test2\Opt-Process-1\SAVE.odb\Opt-Process-1-Job_030'
    # change the test run number below

    # saving the optimised model as a csv
    # save static load results 
    opt_viewport = session.Viewport(name='Optimsation Results Viewport')
    # change the string to an iterative string here:
    opt_Odb_Path = 'H:\ABAQUS\FINALRUN-1\Opt-Process-'+str(i+1)+'\TOSCA_POST\Opt-Process-'+str(i+1)+'-Job_post.odb'
    an_odb_object = session.openOdb(name=opt_Odb_Path)
    opt_viewport.setValues(displayedObject=an_odb_object)

    session.viewports['Viewport: 1'].setValues(displayedObject=an_odb_object)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1)
    report_name_and_path = 'ReportOptResults'+str(i+1)+'.csv'
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)

    # this worked!!! - Output is the material probability
    session.writeFieldReport(fileName=report_name_and_path, append=OFF,sortItem='Element Label', 
                             odb=an_odb_object, step=0, frame=1,outputPosition=NODAL, 
                             variable=(('MAT_PROP_NORMALIZED', ELEMENT_CENTROID, (() )), ))




    opt_viewport = session.Viewport(name='Optimsation Results Viewport')
    # change the string to an iterative string here:
    opt_Odb_Path = 'H:\ABAQUS\FINALRUN-1\Opt-Process-'+str(i+1)+'\TOSCA_POST\Opt-Process-'+str(i+1)+'-Job_post.odb'
    an_odb_object = session.openOdb(name=opt_Odb_Path)
    opt_viewport.setValues(displayedObject=an_odb_object)

    session.viewports['Viewport: 1'].setValues(displayedObject=an_odb_object)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1)
    report_name_and_path = 'ReportOptStressResults'+str(i+1)+'.csv'
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)

    # to save the stress values
    session.writeFieldReport(fileName=report_name_and_path, append=OFF,sortItem='Element Label', 
                             odb=an_odb_object, step=0, frame=1,outputPosition=NODAL, 
                             variable=(('S', INTEGRATION_POINT, ((INVARIANT, 'Mises'), )), ))



    # Save by ctm3017 on 2022_01_29-21.09.03; build 2021 2020_03_06-14.50.37 167380


# END OF MAIN LOOP


    files_ext = ['.jnl','.inp','.res','.lck','.dat','.msg','.sta','.fil','.sim',
              '.stt','.mdl','.prt','.ipm','.log','.com','.odb_f']

    job_name = 'Static-Run-'+str(i+1)

    for file_ex in files_ext:
        file_path = job_name + file_ex
        if os.path.exists(file_path):
            os.remove(file_path)

    opt1_name = 'Opt-Process-'+str(i+1)

    for file_ex in files_ext:
        file_path = opt1_name + file_ex
        if os.path.exists(file_path):
            os.remove(file_path)
    
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.dat')
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.inp')
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.msg')
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.odb')
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.onf')
#     os.rmdir('Opt-Process-'+str(i+1)+'\SAVE.sta')
    
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.dat')
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.inp')
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.msg')
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.odb')
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.onf')
    shutil.rmtree('Opt-Process-'+str(i+1)+'\SAVE.sta')
        
    opt2_name = 'Opt-Process-'+str(i+1)+'-Job'

    for file_ex in files_ext:
        file_path = opt2_name + file_ex
        if os.path.exists(file_path):
            os.remove(file_path)


# In[ ]:





# In[ ]:




