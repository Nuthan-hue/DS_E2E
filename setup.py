from setuptools import setup, find_packages

def get_requirements(file_path)->list:
    #"""Returns a list of requirements from the given file path."""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    
    requirements = [req.strip() for req in requirements if req.strip()]
    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements

 
setup(
    name='DS_E2E',
    version='0.1.0',
    description='A AMchine Learning based Python Project',
    author='Nuthan Kishore Maddineni',
    author_email='nuthan.maddineni23@gmail.com',
    packages=find_packages(),
    #install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'],
    install_requires=get_requirements('requirements.txt'),
)