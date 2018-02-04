from distutils.core import setup

setup(
    name='TwitMine',
    version='0.2dev',
    author='Manezki',
    packages=['twitmine'],
    license='Apache 2.0',
    long_description=open('README.md').read(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib'
    ]
)