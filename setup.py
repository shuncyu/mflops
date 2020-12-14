from setuptools import find_packages, setup

readme = open('README.md').read()

VERSION = '0.1'

requirements = [
    'torch'
]

setup(
    # Metadata
    name='mflops',
    version=VERSION,
    author='Luke Yu',
    author_email='shuncyu@163.com',
    url='https://github.com/shuncyu/mflops',
    description='Model computation for convolutional networks in'
                'pytorch framework',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)
