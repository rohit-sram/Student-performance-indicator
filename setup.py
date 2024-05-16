from setuptools import find_packages, setup
from typing import List

hyph_e = "-e ."
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [r.replace("\n", "") for r in requirements]
            
            if hyph_e in requirements:
                requirements.remove(hyph_e)
                
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Requirements have not been installed.")
                    
    return requirements


setup(
    name="MLProject",
    version='0.0.1',
    author="Rohit Sriram",
    author_email="rohitsram10@outlook.com", 
    packages=find_packages(), 
    # install_requires=['pandas', 'numpy', 'seaborn']  # can be changed depending on the packages we require
    install_requires=get_requirements("requirements.txt")
)