Web App for project "Data Mining the Water Table"


Steps to create and build package:

1. create folder where we store our app data
2. open terminal in the folder and open vscode using 'code .'
3. create new vitual environment using command 'conda create -p venv python==3.9 -y'
4. create README.md file
5. create repository on github
6. use following commands to setup the repository

    git init
    git add README.md
    git commit -m "first commit"
    git branch -M main
    git remote add origin https://github.com/vikrantmohite19/Data_Mining_Water_Table-Web_App.git
    git push -u origin main

7. create setup.py and requirements.txt
8. create source folder as 'DMWT_Package' and a file inside it as '__init__.py' to consider 'DMWT_Package' as a package. 
9. run command 'python -m pip install -r requirements.txt'. (ugrade pip if required using 'python -m pip install --upgrade pip'). this stage will generate a folder 'DMWT_Package.egg-info' 
10. created exception.py and logger.py
11. Created new folder 'notebook' and copied all ipython files into the folder. create data folder inside 'notebook' and copy all raw data files.
<!-- 12. Then to run ipython notebooks , had to run the command in terminal "conda install ipykernel --update-deps --force-reinstall" to install ipykernal. -->
13. Inside the folder 'DMWT_Package' created exception.py, logger.py, utils.py
14. Inside 'DMWT_Package', creat new folder 'pipeline'. Inside 'pipeline created __init__.py, predict_pipeline.py & train_pipeline.py.
15. Inside 'DMWT_Package', create new folder 'components'. Inside 'components' created __init__.py, data_ingestion.py, data_transformation.py, model_trainer.py.
16. Run Data_ingestion.py which inturn will create folders such as artifacts, preprocessed, logs. In this stage we will get preprocessed data and the best model after performing grid-search
17. Run command,  'python setup.py sdist bdist_wheel'. In this stage two folders such as 'build' and 'dist' will be created.
18. Now go to folder 'dist' and run 'pip install DMWT_Package-0.0.1-py3-none-any.whl. this is to test if package can be installed & binary distribution file run properly. 
19. Now run python, run 'import DMWT_Package' and their submodules to check if the same can be imported. 
20. Commit and push everything on git.

