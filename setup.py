from setuptools import find_packages, setup
from typing import List
import os

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='DMWT_Package',
    version='0.0.1',
    author='Vikrant',
    author_email='vikrantmohite9029@gmail.com',
    packages=find_packages(),
    # packages=find_packages() + ['DMWT_Web-App'],
    install_requires=get_requirements("requirements.txt"),
    # DMWT_Web-App={
        
    #     'DMWT_Package': ['requirements.txt', 'README.md', 'app.py', 'setup.py'],
    #     'artifacts': ['*'],
    #     'notebook': ['*'],
    #     'preprocessed': ['*'],
    #     'templates': ['*']

    #     }
)

