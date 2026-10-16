from glob import glob
import os

from setuptools import find_packages, setup

package_name = "wodowx_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Project Maintainer",
    maintainer_email="maintainer@example.com",
    description="Azure Kinect leaf perception and WidowX250s EE measurement pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "leaf_perception = wodowx_control.perception_node:main",
            "leaf_control = wodowx_control.control_node:main",
            "leaf_workflow = wodowx_control.main_node:main",
        ],
    },
)
