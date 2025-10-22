from setuptools import setup, find_packages

setup(
    name="S2ST_Project",
    version="0.1.0",
    packages=find_packages(),
    author="Your Name",
    description="An English-to-Vietnamese Speech-to-Speech Translation System",
    install_requires=[
        'torch',
        'torchaudio',
        'transformers',
        'flask',
        'werkzeug',
        'tqdm',
        'streamlit',
        'pandas',
        'pyinstaller'
    ],
    entry_points={
        'console_scripts': [
            'run_s2st_webapp=run_webapp:main',
        ],
    },
)