import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'antiuav_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Tell colcon to copy your launch files during the build:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bninaos',
    maintainer_email='you@email.com',
    description='Bringup package for Anti-UAV tracking system',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)