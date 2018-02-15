from distutils.core import setup

# TODO Look into possibility of automated Torch install
setup(
    name='TwitMine',
    url='https://github.com/Manezki/TwitMine',
    version='0.2dev',
    author='Manezki',
    packages=['twitmine', 'twitmine/neural_networks'],
    license='Apache 2.0',
    long_description=open('README.md').read(),
    install_requires=[
        'torch',
        'numpy',
    ],
    package_data={
        'twitmine': ["../assets/embeddings/reactionrnn_vocab.json",
                     "../assets/weights/RNN.pt"]
    },
    include_package_data=True,
)