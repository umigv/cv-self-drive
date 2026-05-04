from setuptools import setup

package_name = "cv_self_drive"

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    py_modules=[
        "functional_tests_occ_grid",
        "right_turn",
        "left_turn",
        "self_drive_occ_grid",
        "hsv_tune",
        "hsv",
        "get_cam_idx",
        "plane",
        "zed_hsv_demo"
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="UMARV",
    description="CV selfdrive stack",

    entry_points={
        "console_scripts": [
            "occ_grid_node = functional_tests_occ_grid:main",
        ],
    },
)