from setuptools import find_packages, setup
from glob import glob

package_name = 'vehicle_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh') + glob('scripts/*.py')),
        ('share/' + package_name + '/data', glob('data/*.mat')),
        (
            'share/' + package_name + '/data/reference_velocity',
            glob('data/reference_velocity/*.mat'),
        ),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='yuxuan',
    maintainer_email='yuxuan.shao.300@gmail.com',
    description='ROS 2 vehicle controller with OxTS state input and MPC/LQR control',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vehicle_controller_node = vehicle_controller.vehicle_controller_node:main',
        ],
    },
)
