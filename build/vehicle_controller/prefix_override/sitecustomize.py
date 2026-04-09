import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yuxuan/ros_ws/src/vehicle_controller/install/vehicle_controller'
